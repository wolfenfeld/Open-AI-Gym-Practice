from Modules.Games import DQNCartPoleGame, LunarLanderGame

if __name__ == "__main__":
    # run_cart_pole_game_dqn()

    dqn_cart_pole_game = DQNCartPoleGame()
    dqn_cart_pole_game.run()

    lunar_lander_game = LunarLanderGame()
    lunar_lander_game.run()
