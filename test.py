import pybullet_envs.bullet.minitaur_gym_env as e
env = e.MinitaurBulletEnv(render=True)
ion= env.reset()
env.render()