class SlidingWindowVolumeManager:
    """
    Manages sliding window volume loading for efficient time series extraction
    """

    def __init__(self, base_dir, timeframe):
        self.base_dir = base_dir
        self.timeframe = timeframe
        self.volume_queue = deque()  # (frame_number, registered_volume, label_volume)
        self.current_center_frame = None

    def load_initial_window(self, center_frame):
        """Load initial window of volumes centered on the given frame"""
        self.current_center_frame = center_frame
        frames_to_load = range(
            center_frame - self.timeframe, center_frame + self.timeframe + 1
        )

        print(f"📥 Loading initial window: frames {list(frames_to_load)}")

        for frame in frames_to_load:
            volume_data = get_volume_by_timestamp(self.base_dir, frame)
            if (
                volume_data["registered_image"] is not None
                and volume_data["label_image"] is not None
            ):
                self.volume_queue.append(
                    (frame, volume_data["registered_image"], volume_data["label_image"])
                )
                print(f"  ✅ Loaded frame {frame}")
            else:
                self.volume_queue.append((frame, None, None))
                print(f"  ❌ Failed to load frame {frame}")

    def slide_to_frame(self, new_center_frame):
        """Slide the window to center on a new frame"""
        if self.current_center_frame is None:
            self.load_initial_window(new_center_frame)
            return

        frame_shift = new_center_frame - self.current_center_frame
        if frame_shift == 0:
            return  # Already at the right position

        print(
            f"🔄 Sliding window from {self.current_center_frame} to {new_center_frame} (shift: {frame_shift})"
        )

        if abs(frame_shift) >= len(self.volume_queue):
            # Complete reload needed
            self.volume_queue.clear()
            self.load_initial_window(new_center_frame)
            return

        # Incremental slide
        if frame_shift > 0:
            # Moving forward - remove from left, add to right
            for _ in range(frame_shift):
                self.volume_queue.popleft()

            # Add new frames to the right
            start_frame = self.current_center_frame + self.timeframe + 1
            for i in range(frame_shift):
                frame = start_frame + i
                volume_data = get_volume_by_timestamp(self.base_dir, frame)
                if (
                    volume_data["registered_image"] is not None
                    and volume_data["label_image"] is not None
                ):
                    self.volume_queue.append(
                        (
                            frame,
                            volume_data["registered_image"],
                            volume_data["label_image"],
                        )
                    )
                else:
                    self.volume_queue.append((frame, None, None))

        else:
            # Moving backward - remove from right, add to left
            for _ in range(-frame_shift):
                self.volume_queue.pop()

            # Add new frames to the left
            start_frame = self.current_center_frame - self.timeframe - 1
            for i in range(-frame_shift):
                frame = start_frame - i
                volume_data = get_volume_by_timestamp(self.base_dir, frame)
                if (
                    volume_data["registered_image"] is not None
                    and volume_data["label_image"] is not None
                ):
                    self.volume_queue.appendleft(
                        (
                            frame,
                            volume_data["registered_image"],
                            volume_data["label_image"],
                        )
                    )
                else:
                    self.volume_queue.appendleft((frame, None, None))

        self.current_center_frame = new_center_frame

    def get_volumes_for_extraction(self):
        """Get all volumes in current window for extraction"""
        return list(self.volume_queue)
