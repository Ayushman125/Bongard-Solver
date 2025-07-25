class GAGANGenerator:
    @staticmethod
    def load(problem_id: str) -> "GAGANGenerator":
        # load GAN weights per problem
        return GAGANGenerator()

    def sample_latents(self, n: int):
        # return list of latent vectors
        return [self._random_latent() for _ in range(n)]

    def generate_from_latents(self, latents):
        # map each latent to a LOGO action_program
        return [self._decode(lat) for lat in latents]

    def _random_latent(self):
        # stub: return a random latent vector
        import numpy as np
        return np.random.randn(128)

    def _decode(self, lat):
        # stub: decode latent to action_program
        return [("line", 0.5), ("rotate", 30)]
