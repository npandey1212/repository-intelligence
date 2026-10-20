"""Service helpers used by construction-rule fixtures."""


class Service:
    def run(self):
        return True


def run():
    return Service().run()
