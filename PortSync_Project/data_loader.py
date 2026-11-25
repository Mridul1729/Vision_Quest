import json
import os

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")


def load_json(filename):
    path = os.path.join(DATA_FOLDER, filename)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_ships():
    return load_json("ships.json")


def load_berths():
    return load_json("berths.json")


def load_cranes():
    return load_json("cranes.json")


def load_containers():
    return load_json("containers.json")


def load_vehicles():
    return load_json("vehicles.json")


def load_drivers():
    return load_json("drivers.json")


def load_warehouses():
    return load_json("warehouses.json")


def load_dispatch():
    return load_json("dispatch.json")


def load_all_data():
    return {
        "ships": load_ships(),
        "berths": load_berths(),
        "cranes": load_cranes(),
        "containers": load_containers(),
        "vehicles": load_vehicles(),
        "drivers": load_drivers(),
        "warehouses": load_warehouses(),
        "dispatch": load_dispatch(),
    }


if __name__ == "__main__":
    data = load_all_data()
    for name, items in data.items():
        print(f"\n{name.upper()}:")
        print(items)
