import json

class Metrics:
    def __init__(self, id) -> None:
        super().__setattr__("content", {})  # Evita erro de recursão
        super().__setattr__("id", id)

    def store(self, key: str, value: str) -> None:
        self.content[key] = value

    def append(self, key: str, value: str) -> None:
        if key not in self.content:
            self.content[key] = []
        self.content[key].append(value)

    def access(self, keys: list, value: str) -> None:
        head = self.content
        for key in keys[:-1]:  # Percorre os intermediários
            if key not in head:
                head[key] = {}
            head = head[key]  # Atualiza a referência para o próximo nível
        head[keys[-1]] = [value]  # Adiciona o valor final

    def __getattr__(self, name):
        return self.content.get(name, None)

    def __setattr__(self, name, value):
        if name in ("content", "id"):  # Evita recursão infinita
            super().__setattr__(name, value)
        else:
            self.content[name] = value

    def __getitem__(self, key):
        return self.content.get(key, None)  # Permite `m['init']`

    def __setitem__(self, key, value):
        self.content[key] = value  # Permite `m['init'] = 'some_value'`

    def __repr__(self):
        return str(self.content)


if __name__ == '__main__':
    m = Metrics(1)
    m["init"] = "initialized"  # Agora funciona
    print(m["init"])  # Deve imprimir 'initialized'

    m.store("key1", "value1")
    print(m["key1"])  # value1

    m.append("list_key", "item1")
    m.append("list_key", "item2")
    print(m["list_key"])  # ['item1', 'item2']

    m.access(["nested", "key"], "deep_value")
    print(m["nested"])  # {'key': ['deep_value']}

    print(m)

