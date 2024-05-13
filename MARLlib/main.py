from marllib import marl

env = marl.make_env(environment_name="mpe", map_name="simple_spread")

mappo = marl.algos.mappo(hyperparam_source='mpe')

model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

mappo.fit(env, model, stop={"timesteps_total": 1000000}, checkpoint_freq=100, share_policy="group")
