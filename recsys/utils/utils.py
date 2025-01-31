
class IDConverter:
    def __init__(self):
        self.id_to_int = {}
        self.int_to_id = {}
        self.counter = 1

    def convert(self, id_str):
        if id_str not in self.id_to_int:
            self.id_to_int[id_str] = self.counter
            self.int_to_id[self.counter] = id_str
            self.counter += 1
        return self.id_to_int[id_str]

    def reverse_convert(self, int_id):
        return self.int_to_id[int_id]
    
