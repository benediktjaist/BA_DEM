
class PositionTracker:
    # particle_position_tracker is ip1 list of lists
    # with the first index corresponding to time and the second index corresponding to particle id
    def __init__(self, particles: List[Particle], simtime: float, dt: float):
        self.particles = particles
        self.simtime = simtime
        self.dt = dt
        self.positions = [[] for _ in range(int(self.simtime / self.dt))]
        self.rotations = [[] for _ in range(int(self.simtime / self.dt))]
        for i in range(len(self.positions)):
            for particle in self.particles:
                self.positions[i].append([])
                self.rotations[i].append([])

    def update(self, t: int):
        for i, particle in enumerate(self.particles):
            self.positions[t][i].append(particle.position)
            self.rotations[t][i].append(particle.rotation)