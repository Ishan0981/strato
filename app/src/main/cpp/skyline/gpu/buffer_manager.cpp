// SPDX-License-Identifier: MPL-2.0
// Copyright © 2021 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include <common/settings.h>
#include <gpu.h>
#include "buffer_manager.h"

namespace skyline::gpu {
    BufferManager::BufferManager(GPU &gpu) : gpu{gpu} {}

    bool BufferManager::BufferLessThan(const std::shared_ptr<Buffer> &it, u8 *pointer) {
        return it->guest->begin().base() < pointer;
    }

    BufferManager::LockedBuffer::LockedBuffer(std::shared_ptr<Buffer> pBuffer, ContextTag tag) : buffer{std::move(pBuffer)}, lock{tag, *buffer}, stateLock(buffer->stateMutex) {}

    Buffer *BufferManager::LockedBuffer::operator->() const {
        return buffer.get();
    }

    std::shared_ptr<Buffer> &BufferManager::LockedBuffer::operator*() {
        return buffer;
    }

    BufferManager::LockedBuffers BufferManager::Lookup(span<u8> range, ContextTag tag) {
        LockedBuffers overlaps;

        // Try to do a fast lookup in the page table
        auto lookupBuffer{bufferTable[range.begin().base()]};
        if (lookupBuffer != nullptr && lookupBuffer->guest->contains(range)) {
            overlaps.emplace_back(lookupBuffer->shared_from_this(), tag);
            return overlaps;
        }

        // If we cannot find the buffer quickly, do a binary search to find all overlapping buffers
        auto entryIt{std::lower_bound(bufferMappings.begin(), bufferMappings.end(), range.end().base(), BufferLessThan)};
        while (entryIt != bufferMappings.begin() && (*--entryIt)->guest->begin() <= range.end())
            if ((*entryIt)->guest->end() > range.begin())
                overlaps.emplace_back(*entryIt, tag);

        return overlaps;
    }
     

void BufferManager::InsertBuffer(std::shared_ptr<Buffer> buffer) {
    auto bufferStart{buffer->guest->begin().base()}, bufferEnd{buffer->guest->end().base()};
    bufferTable.Set(bufferStart, bufferEnd, buffer.get());
    auto insertPosition = std::lower_bound(bufferMappings.begin(), bufferMappings.end(), bufferEnd, BufferLessThan);
    bufferMappings.emplace(insertPosition, std::move(buffer));
}

void BufferManager::DeleteBuffer(const std::shared_ptr<Buffer> &buffer) {
    bufferTable.Set(buffer->guest->begin().base(), buffer->guest->end().base(), nullptr);
    auto bufferIterator = std::find(bufferMappings.begin(), bufferMappings.end(), buffer);
    if (bufferIterator != bufferMappings.end()) {
        bufferMappings.erase(bufferIterator);
    }
}

BufferManager::LockedBuffer BufferManager::CoalesceBuffers(span<u8> range, const LockedBuffers &srcBuffers, ContextTag tag) {
    TRACE_EVENT("gpu", "BufferManager::CoalesceBuffers");
    std::shared_ptr<FenceCycle> newBufferCycle{};
    for (auto &srcBuffer : srcBuffers) {
        if (!*gpu.state.settings->useDirectMemoryImport && (srcBuffer->dirtyState == Buffer::DirtyState::GpuDirty || srcBuffer->AllCpuBackingWritesBlocked()))
            srcBuffer->WaitOnFence();

        if (newBufferCycle && srcBuffer->cycle != newBufferCycle)
            srcBuffer->WaitOnFence();
        else
            newBufferCycle = srcBuffer->cycle;
    }

    std::scoped_lock lock{recreationMutex};
    if (!range.valid())
        range = span<u8>{srcBuffers.front().buffer->guest->begin(), srcBuffers.back().buffer->guest->end()};

    auto lowestAddress{range.begin().base()}, highestAddress{range.end().base()};
    for (const auto &srcBuffer : srcBuffers) {
        auto mapping{*srcBuffer->guest};
        if (mapping.begin().base() < lowestAddress)
            lowestAddress = mapping.begin().base();
        if (mapping.end().base() > highestAddress)
            highestAddress = mapping.end().base();
    }

    bufferMappings.emplace_back(std::make_shared<Buffer>(delegateAllocatorState, gpu, span<u8>{lowestAddress, highestAddress}, nextBufferId++, *gpu.state.settings->useDirectMemoryImport), tag);
    auto& newBuffer = bufferMappings.back();
    
    newBuffer->SetupStagedTraps();
    newBuffer->SynchronizeHost(false);
    newBuffer->cycle = newBufferCycle;

    auto copyBuffer = [](auto dstGuest, auto srcGuest, auto dstPtr, auto srcPtr) {
        if (dstGuest.begin().base() <= srcGuest.begin().base()) {
            size_t dstOffset{static_cast<size_t>(srcGuest.begin().base() - dstGuest.begin().base())};
            size_t copySize{std::min(dstGuest.size() - dstOffset, srcGuest.size())};
            std::memcpy(dstPtr + dstOffset, srcPtr, copySize);
        } else if (dstGuest.begin().base() > srcGuest.begin().base()) {
            size_t srcOffset{static_cast<size_t>(dstGuest.begin().base() - srcGuest.begin().base())};
            size_t copySize{std::min(dstGuest.size(), srcGuest.size() - srcOffset)};
            std::memcpy(dstPtr, srcPtr + srcOffset, copySize);
        }
    };

    for (auto &srcBuffer : srcBuffers) {
        if (newBuffer->backingImmutability == Buffer::BackingImmutability::None && srcBuffer->backingImmutability != Buffer::BackingImmutability::None)
            newBuffer->backingImmutability = srcBuffer->backingImmutability;
        else if (srcBuffer->backingImmutability == Buffer::BackingImmutability::AllWrites)
            newBuffer->backingImmutability = Buffer::BackingImmutability::AllWrites;

        newBuffer->everHadInlineUpdate |= srcBuffer->everHadInlineUpdate;

        if (!*gpu.state.settings->useDirectMemoryImport) {
            if (srcBuffer->dirtyState == Buffer::DirtyState::GpuDirty) {
                if (srcBuffer.lock.IsFirstUsage() && newBuffer->dirtyState != Buffer::DirtyState::GpuDirty)
                    copyBuffer(*newBuffer->guest, *srcBuffer->guest, newBuffer->mirror.data(), srcBuffer->backing->data());
                else
                    newBuffer->MarkGpuDirtyImpl();

                copyBuffer(*newBuffer->guest, *srcBuffer->guest, newBuffer->backing->data(), srcBuffer->backing->data());
            } else if (srcBuffer->AllCpuBackingWritesBlocked()) {
                if (srcBuffer->dirtyState == Buffer::DirtyState::CpuDirty)
                    LOGE("Buffer (0x{}-0x{}) is marked as CPU dirty while CPU backing writes are blocked, this is not valid");

                copyBuffer(*newBuffer->guest, *srcBuffer->guest, newBuffer->backing->data(), srcBuffer->backing->data());
            }
        } else {
            if (srcBuffer->RefreshGpuWritesActiveDirect(false, {})) {
                newBuffer->MarkGpuDirtyImpl();
            } else if (srcBuffer->directTrackedShadowActive) {
                newBuffer->EnableTrackedShadowDirect();
                copyBuffer(*newBuffer->guest, *srcBuffer->guest, newBuffer->directTrackedShadow.data(), srcBuffer->directTrackedShadow.data());
                newBuffer->directTrackedWrites.Merge(srcBuffer->directTrackedWrites);
            }
        }

        vk::DeviceSize overlapOffset{static_cast<vk::DeviceSize>(srcBuffer->guest->begin() - newBuffer->guest->begin())};
        srcBuffer->delegate->Link(newBuffer->delegate, overlapOffset);
    }

    return newBuffer;
}


    }

    BufferView BufferManager::FindOrCreateImpl(GuestBuffer guestMapping, ContextTag tag, const std::function<void(std::shared_ptr<Buffer>, ContextLock<Buffer> &&)> &attachBuffer) {
        /*
         * We align the buffer to the page boundary to ensure that:
         * 1) Any buffer view has the same alignment guarantees as on the guest, this is required for UBOs, SSBOs and Texel buffers
         * 2) We can coalesce a lot of tiny buffers into a single large buffer covering an entire page, this is often the case for index buffers and vertex buffers
         */
        auto alignedStart{util::AlignDown(guestMapping.begin().base(), constant::PageSize)}, alignedEnd{util::AlignUp(guestMapping.end().base(), constant::PageSize)};
        span<u8> alignedGuestMapping{alignedStart, alignedEnd};

        auto overlaps{Lookup(alignedGuestMapping, tag)};
        if (overlaps.size() == 1) [[likely]] {
            // If we find a buffer which can entirely fit the guest mapping, we can just return a view into it
            auto &firstOverlap{overlaps.front()};
            if (firstOverlap->guest->begin() <= alignedGuestMapping.begin() && firstOverlap->guest->end() >= alignedGuestMapping.end())
                return firstOverlap->GetView(static_cast<vk::DeviceSize>(guestMapping.begin() - firstOverlap->guest->begin()), guestMapping.size());
        }

        if (overlaps.empty()) {
            // If we couldn't find any overlapping buffers, create a new buffer without coalescing
            LockedBuffer buffer{std::make_shared<Buffer>(delegateAllocatorState, gpu, alignedGuestMapping, nextBufferId++, *gpu.state.settings->useDirectMemoryImport), tag};
            buffer->SetupStagedTraps();
            InsertBuffer(*buffer);
            return buffer->GetView(static_cast<vk::DeviceSize>(guestMapping.begin() - buffer->guest->begin()), guestMapping.size());
        } else {
            // If the new buffer overlaps other buffers, we need to create a new buffer and coalesce all overlapping buffers into one
            auto buffer{CoalesceBuffers(alignedGuestMapping, overlaps, tag)};

            // If any overlapping buffer was already attached to the current context, we should also attach the new buffer
            for (auto &srcBuffer : overlaps) {
                if (!srcBuffer.lock.IsFirstUsage()) {
                    attachBuffer(*buffer, std::move(buffer.lock));
                    break;
                }
            }

            // Delete older overlapping buffers and insert the new buffer into the map
            for (auto &overlap : overlaps) {
                DeleteBuffer(*overlap);
                overlap->Invalidate(); // Invalidate the overlapping buffer so it can't be synced in the future
            }
            InsertBuffer(*buffer);

            return buffer->GetView(static_cast<vk::DeviceSize>(guestMapping.begin() - buffer->guest->begin()), guestMapping.size());
        }
    }
}
