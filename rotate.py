import numpy as np
import cv2
import time
import math

class RotateAndCrop:
    def largest_rotated_rect(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.

        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

        Converted to Python by Aaron Snoswell
        """

        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )

    def crop_around_center(self, image, width, height):
        """
        Given a NumPy / OpenCV 2 image, crops it to the given width and height,
        around it's centre point
        """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if(width > image_size[0]):
            width = image_size[0]

        if(height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        # rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        
        scale = 1
        a = scale * math.cos(math.radians(angle))
        b = scale * math.sin(math.radians(angle))
        center = np.zeros(2)
        center[0] = image_center[0]
        center[1] = image_center[1]


        rot_mat = np.zeros((2,3))
        rot_mat[0,0] = a
        rot_mat[0,1] = b
        rot_mat[0,2] = (1-a)*center[0] - b*center[1]
        rot_mat[1,0] = -b
        rot_mat[1,1] = a
        rot_mat[1,2] = b*center[0] + (1-a)*center[1]
        print(rot_mat)


        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result


    def rotate_and_crop(self, img, angle):
        image_height, image_width = img.shape[0:2]
        return self.crop_around_center(self.rotate_image(img, angle), *self.largest_rotated_rect(image_width, image_height, math.radians(angle)))

rnc = RotateAndCrop()
# Using cv2.imread() method
img = cv2.imread("./aaa.jpg")
angle_in_degree = 10

# Displaying the image
cv2.imshow('image', rnc.rotate_and_crop(img, angle_in_degree))
cv2.waitKey(0)
cv2.destroyAllWindows()
