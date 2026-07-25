"""Runtime dependency guards for provider-managed TPU environments."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable, Mapping


TPU_RUNTIME_DISTRIBUTIONS = ("jax", "jaxlib", "libtpu")


def distribution_versions(
    names: Iterable[str] = TPU_RUNTIME_DISTRIBUTIONS,
) -> dict[str, str | None]:
    """Return installed versions without importing native TPU modules."""

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def require_unchanged_tpu_runtime(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> None:
    """Fail before JAX starts if setup replaced provider TPU distributions."""

    changed = {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in TPU_RUNTIME_DISTRIBUTIONS
        if before.get(name) != after.get(name)
    }
    if changed:
        raise RuntimeError(
            "TPU runtime distributions changed during dependency setup: "
            f"{changed}"
        )
