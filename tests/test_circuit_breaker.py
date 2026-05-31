import asyncio
import pytest
from app.core.circuit_breaker import CircuitBreaker, CircuitOpenError


@pytest.mark.asyncio
async def test_closed_allows_calls():
    cb = CircuitBreaker("test", failure_threshold=3)

    async def success():
        return 42

    result = await cb.call(success)
    assert result == 42
    assert cb._state.status == "closed"


@pytest.mark.asyncio
async def test_resets_failure_count_on_success():
    cb = CircuitBreaker("test", failure_threshold=5)

    async def failing():
        raise RuntimeError("fail")

    # 2 failures
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(failing)
    assert cb._state.failure_count == 2

    # 1 success resets count
    async def success():
        return "ok"

    await cb.call(success)
    assert cb._state.failure_count == 0


@pytest.mark.asyncio
async def test_opens_after_threshold():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=60)

    async def failing():
        raise RuntimeError("fail")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(failing)

    assert cb.is_open
    assert cb._state.status == "open"

    # Next call should be circuit-opened
    with pytest.raises(CircuitOpenError):
        await cb.call(failing)


@pytest.mark.asyncio
async def test_half_open_after_timeout():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)

    async def failing():
        raise RuntimeError("fail")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(failing)

    assert cb.is_open

    await asyncio.sleep(0.15)  # Wait for cooldown
    assert not cb.is_open  # Should allow attempt


@pytest.mark.asyncio
async def test_recovers_from_half_open():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1, half_open_max=1)

    async def failing():
        raise RuntimeError("fail")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(failing)

    await asyncio.sleep(0.15)

    async def success():
        return "ok"

    result = await cb.call(success)
    assert result == "ok"
    assert cb._state.status == "closed"


@pytest.mark.asyncio
async def test_half_open_failure_reopens():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1, half_open_max=1)

    async def failing():
        raise RuntimeError("fail")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(failing)

    await asyncio.sleep(0.15)

    # Half-open probe fails
    with pytest.raises(RuntimeError):
        await cb.call(failing)

    assert cb._state.status == "open"


@pytest.mark.asyncio
async def test_circuit_open_error_not_counted_as_failure():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=60)

    async def failing():
        raise RuntimeError("fail")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(failing)

    initial_count = cb._state.failure_count

    # CircuitOpenError should not increment failure count
    with pytest.raises(CircuitOpenError):
        await cb.call(failing)

    assert cb._state.failure_count == initial_count


def test_reset():
    cb = CircuitBreaker("test", failure_threshold=1)
    cb._state.status = "open"
    cb._state.failure_count = 99
    cb.reset()
    assert cb._state.status == "closed"
    assert cb._state.failure_count == 0
