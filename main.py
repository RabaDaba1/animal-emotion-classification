import multiprocessing

from services.process_manager import ProcessManager

multiprocessing.freeze_support()

if __name__ == "__main__":
    manager = ProcessManager()
    manager.start()

    try:
        manager.view_all_outputs()

        # manager.view_raw_frames()
        # manager.view_detection_results()
        # manager.view_cropped_pets()
    finally:
        manager.stop()
