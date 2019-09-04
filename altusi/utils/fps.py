"""
FPS Measurer class
==================

Class that supports Time and FPS measurement

Revision
--------
    2019, Apr 19: file created
"""

import datetime

class FPS:
    """Time and FPS measurer class"""

    def __init__(self):
        """Initialization for FPS class"""

        self.__start_t = None
        self.__end_t = None
        self.__last_t = None
        self.__no_frames = 0


    def start(self):
        """Start the timer 

        Start to measure time and store the starting moment
        """

        self.__start_t = datetime.datetime.now()


    def stop(self):
        """Stop the timer
        
        Stop to measure time and store the ending moment
        """

        self.__end_t = datetime.datetime.now()
        

    def elapsed(self):
        """Elapsed time measurement

        Measure the elapsed time from the begining to the current invoking time

        Returns:
        --------
            float
                amount of elapsed time in seconds
        """

        curr_t = datetime.datetime.now()
        return (curr_t - self.__start_t).total_seconds()


    def update(self):
        """Update measurer

        Increase the number of processed frames and 
        when the last one is done processing
        """

        self.__last_t = datetime.datetime.now()
        self.__no_frames += 1


    def fps(self):
        """FPS measurement

        Returns:
        --------
            float
                corresponding FPS at the latest processed frame
        """

        processed_t = (self.__last_t - self.__start_t).total_seconds()
        return self.__no_frames / processed_t

    
    def totalTime(self):
        """Total processing time measurement

        Returns:
        --------
            float
                total processing time
        """

        if self.__end_t is None:
            self.stop()
        return (self.__end_t - self.__start_t).total_seconds()
