from models import COLORS

class GlobalIDs:
    def __init__(self):
        self.of = {1: {}, 2: {}}    # cam -> {local_id: global_id}
        self.pair = {}              # global_id -> {1: local_id, 2: local_id}
        self.next_id = 1

    def get(self, cam, tid):
        return self.of[cam].get(tid)

    def allowed(self, t1, t2, act1, act2):
        g1, g2 = self.get(1, t1), self.get(2, t2)
        if g1 is not None and g2 is not None:
            return g1 == g2
        if g1 is not None:
            o = self.pair[g1].get(2)
            return o is None or o == t2 or o not in act2
        if g2 is not None:
            o = self.pair[g2].get(1)
            return o is None or o == t1 or o not in act1
        return True

    def link(self, t1, t2):
        g1, g2 = self.get(1, t1), self.get(2, t2)
        if g1 is not None and g1 == g2:
            return
        if g1 is not None:
            self.of[2][t2] = g1
            self.pair[g1][2] = t2
        elif g2 is not None:
            self.of[1][t1] = g2
            self.pair[g2][1] = t1
        else:
            g = self.next_id
            self.next_id += 1
            self.of[1][t1] = g
            self.of[2][t2] = g
            self.pair[g] = {1: t1, 2: t2}


def draw(frame, box, gid, text):
    color = COLORS[(gid - 1) % len(COLORS)]
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
