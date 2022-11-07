particle_list = ["p1", "p2", "p3"]
# newlist = [x for x in particle_list if "p" in x]
# print(newlist)
# exit()


class Particle:
    list_of_particles = []

    def __init__(self, position):
        self.position = position
        Particle.list_of_particles.append(self)


p1 = Particle([0, 0])
p2 = Particle([1, 1])
p3 = Particle([2, 2])
'''
for index1,partikel_i in enumerate(particle_list): # i = counter j = partikel
    for index2, partikel_j in enumerate(particle_list):
        if index2>index1:
            pass
        else:
            pass
            
for index,partikel_i in enumerate(particle_list):
    for partikel_j in particle_list[index+1:]:      # list indexing
        print(partikel_i,partikel_j)
'''
for index, pi in enumerate(Particle.list_of_particles):
    for pj in Particle.list_of_particles[index + 1:]:
        if index == 1:
            pj.position = [4, 4]
            print(Particle.list_of_particles[2].position)
