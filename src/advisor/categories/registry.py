"""Category registration interfaces without category-specific rules."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from pydantic import BaseModel

from advisor.categories.base import CategorySpec


class CategoryDefinition(BaseModel):
    """Location metadata for an independently implemented category."""

    name: str
    display_name: str = ""
    package_path: str
    config_path: Path
    implemented: bool = True
    factory_name: str = "get_category_spec"


class CategoryRegistry:
    """In-memory registry for category modules."""

    def __init__(self) -> None:
        self._categories: dict[str, CategoryDefinition] = {}
        self._specs: dict[str, CategorySpec] = {}

    def register(self, definition: CategoryDefinition) -> None:
        """Register or replace one category definition."""
        self._categories[definition.name] = definition
        self._specs.pop(definition.name, None)

    def register_spec(
        self, definition: CategoryDefinition, spec: CategorySpec
    ) -> None:
        """Register an eagerly constructed spec, primarily for tests/embedding."""
        spec.validate()
        if definition.name != spec.name:
            raise ValueError("Category definition and spec names must match")
        self._categories[definition.name] = definition
        self._specs[definition.name] = spec

    def get(self, name: str) -> CategoryDefinition:
        """Return a category definition by name."""
        return self._categories[name]

    def all(self) -> dict[str, CategoryDefinition]:
        """Return a shallow copy of registered category definitions."""
        return self._categories.copy()

    def get_spec(self, name: str) -> CategorySpec:
        """Lazily load and validate a category's behavior contract."""
        if name in self._specs:
            return self._specs[name]
        definition = self.get(name)
        if not definition.implemented:
            raise KeyError(f"Category {name!r} is not implemented")
        module = import_module(definition.package_path)
        factory = getattr(module, definition.factory_name, None)
        if not callable(factory):
            raise ValueError(
                f"{definition.package_path} must export {definition.factory_name}()"
            )
        spec = factory()
        if not isinstance(spec, CategorySpec):
            raise TypeError(
                f"{definition.package_path}.{definition.factory_name}() must return "
                "CategorySpec"
            )
        spec.validate()
        if spec.name != definition.name:
            raise ValueError("Category definition and spec names must match")
        self._specs[name] = spec
        return spec

    def validate_all(self) -> None:
        """Validate enabled specs and reject collection ownership conflicts."""
        collections: dict[str, str] = {}
        for name, definition in self._categories.items():
            if not definition.implemented:
                continue
            spec = self.get_spec(name)
            collection = str(spec.config["collection"])
            owner = collections.get(collection)
            if owner is not None:
                raise ValueError(
                    f"Qdrant collection {collection!r} is shared by categories "
                    f"{owner!r} and {name!r}"
                )
            collections[collection] = name


def build_default_registry() -> CategoryRegistry:
    """Register all implemented product categories."""
    registry = CategoryRegistry()
    registry.register(
        CategoryDefinition(
            name="refrigerator",
            display_name="Tủ Lạnh",
            package_path="advisor.categories.refrigerator",
            config_path=Path(__file__).with_name("refrigerator") / "config.yaml",
        )
    )
    registry.register(
        CategoryDefinition(
            name="air_conditioner",
            display_name="Máy lạnh",
            package_path="advisor.categories.air_conditioner",
            config_path=Path(__file__).with_name("air_conditioner") / "config.yaml",
        )
    )
    registry.register(
        CategoryDefinition(
            name="washing_machine",
            display_name="Máy giặt",
            package_path="advisor.categories.washing_machine",
            config_path=Path(__file__).with_name("washing_machine") / "config.yaml",
        )
    )
    registry.register(
        CategoryDefinition(
            name="dryer",
            display_name="Máy sấy quần áo",
            package_path="advisor.categories.dryer",
            config_path=Path(__file__).with_name("dryer") / "config.yaml",
        )
    )
    registry.register(
        CategoryDefinition(
            name="dishwasher",
            display_name="Máy rửa chén",
            package_path="advisor.categories.dishwasher",
            config_path=Path(__file__).with_name("dishwasher") / "config.yaml",
        )
    )
    registry.register(
        CategoryDefinition(
            name="cooler_freezer",
            display_name="Tủ mát, tủ đông",
            package_path="advisor.categories.cooler_freezer",
            config_path=Path(__file__).with_name("cooler_freezer") / "config.yaml",
        )
    )
    registry.register(
        CategoryDefinition(
            name="water_heater",
            display_name="Máy nước nóng",
            package_path="advisor.categories.water_heater",
            config_path=Path(__file__).with_name("water_heater") / "config.yaml",
        )
    )
    registry.register(
        CategoryDefinition(
            name="karaoke_microphone",
            display_name="Micro karaoke",
            package_path="advisor.categories.karaoke_microphone",
            config_path=Path(__file__).with_name("karaoke_microphone")
            / "config.yaml",
        )
    )
    registry.register(
        CategoryDefinition(
            name="smartwatch",
            display_name="Đồng hồ thông minh",
            package_path="advisor.categories.smartwatch",
            config_path=Path(__file__).with_name("smartwatch") / "config.yaml",
        )
    )
    registry.register(
        CategoryDefinition(
            name="tablet",
            display_name="Máy tính bảng",
            package_path="advisor.categories.tablet",
            config_path=Path(__file__).with_name("tablet") / "config.yaml",
        )
    )
    registry.register(
        CategoryDefinition(
            name="printer",
            display_name="Máy in",
            package_path="advisor.categories.printer",
            config_path=Path(__file__).with_name("printer") / "config.yaml",
        )
    )
    return registry
