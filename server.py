from openreward.environments import Server

from icu_coordinator import ICUCoordinatorEnvironment

if __name__ == "__main__":
    server = Server([ICUCoordinatorEnvironment])
    server.run()
