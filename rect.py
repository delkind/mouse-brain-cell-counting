from collections import namedtuple


class Rect(namedtuple("RectBase", ['x', 'y', 'w', 'h'])):
    def point_in(self, x, y):
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def rect_in(self, r):
        return self.point_in(*r.corners()[0]) and self.point_in(*r.corners()[1])

    def intersection(self, rect):
        x = max(self.x, rect.x)
        y = max(self.y, rect.y)
        w = min(self.x + self.w, rect.x + rect.w) - x
        h = min(self.y + self.h, rect.y + rect.h) - y
        if w < 0 or h < 0:
            return Rect(0, 0, 0, 0)

        return Rect(x, y, w, h)
    __mul__ = intersection

    def area(self):
        return self.h * self.w

    def sub_frame(self, frame):
        return frame[self.y:self.y + self.h, self.x:self.x + self.w]

    def at_border(self, frame):
        return self.x == 0 or self.x + self.w == frame.shape[0] or self.y == 0 or self.y + self.h == frame.shape[1]

    def pad(self, padding_x, padding_y):
        return Rect(self.x - padding_x, self.y - padding_y, self.w + padding_x, self.h + padding_y)

    def corners(self):
        return (self.x, self.y), (self.x + self.w, self.y + self.h)

    def union(self, rect):
        if rect == self:
            return self

        x = min(self.x, rect.x)
        y = min(self.y, rect.y)
        w = max(self.x + self.w, rect.x + rect.w) - x
        h = max(self.y + self.h, rect.y + rect.h) - y

        return Rect(x, y, w, h)
    __add__ = union

    def scale(self, factor):
        return Rect(y=int(self.y * factor), x=int(self.x * factor), w=int(self.w * factor), h=int(self.h * factor))

    def move(self, offset):
        return Rect(y=int(self.y + offset), x=int(self.x + offset), w=self.w, h=self.h)

    @staticmethod
    def from_center(p, w, h):
        return Rect(y=p.y - h // 2, x=p.x - w // 2, h=h, w=w)
