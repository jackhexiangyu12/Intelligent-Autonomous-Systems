#Daniel D. Lee, Alex Kushleyev, Kelsey Saulnier, Nikolay Atanasov
cimport cython
import numpy as np

#Bresenham's line algorithm
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def getMapCellsFromRay_fclad(int xrobot, int yrobot,
                             xends,yends, 
                             int maxMap):
        cdef int nPoints, index
        cdef int x, y
        cdef int x_iterator
        cdef int x1, y1, temp
        cdef int x0, y0
        cdef float error
        cdef int deltax, deltay
        cdef int ystep
        cdef int steep
        nPoints = np.size(xends)
        index = 0
        lineMap = np.zeros([maxMap * nPoints, 2], dtype=np.int16)
        cdef short [:, :] lineMap_view = lineMap
        cdef short [:] xis_view = xends
        cdef short [:] yis_view = yends
        for idx in range(nPoints):
                x1 = xis_view[idx]
                y1 = yis_view[idx]
                x0 = xrobot 
                y0 = yrobot
                steep = (abs(y1-y0) > abs(x1-x0))
                if steep:
                        temp = x0
                        x0 = y0
                        y0 = temp
                        temp = x1
                        x1 = y1
                        y1 = temp
                if x0 > x1:
                        temp = x0
                        x0 = x1
                        x1 = temp
                        temp = y0
                        y0 = y1
                        y1 = temp
                deltax = x1 - x0
                deltay = abs(y1 - y0)
                error = deltax / 2.
                y = y0
                ystep = 0
                if y0 < y1:
                  ystep = 1
                else:
                  ystep = -1
                x_iterator = x0
                if steep:
                        while(x_iterator < x1):
                                lineMap_view[index, 0] = y
                                lineMap_view[index, 1] = x_iterator
                                index += 1
                                error = error - deltay
                                if error < 0:
                                        y += ystep
                                        error += deltax
                                x_iterator += 1
                else:
                        while(x_iterator < x1):
                                lineMap_view[index, 0] = x_iterator
                                lineMap_view[index, 1] = y
                                index += 1
                                error = error - deltay
                                if error < 0:
                                        y += ystep
                                        error += deltax
                                x_iterator += 1
        xyio = lineMap[:index, :].T
        return xyio
