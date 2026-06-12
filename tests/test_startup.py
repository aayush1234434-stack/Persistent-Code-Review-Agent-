from unittest.mock import MagicMock, patch

import pytest

import agent
import main


def test_stategraph_available_when_langgraph_installed():
    assert agent.StateGraph is not None
    assert agent.END is not None


def test_init_langgraph_persistence_uses_pool_not_context_manager():
    fake_pool = MagicMock(name="pool")
    fake_checkpointer = MagicMock(name="checkpointer")
    fake_store = MagicMock(name="store")
    fake_saver_cls = MagicMock(return_value=fake_checkpointer)
    fake_store_cls = MagicMock(return_value=fake_store)

    with patch.object(main, "create_langgraph_pg_pool", return_value=fake_pool) as create_pool:
        with patch.object(main.importlib, "import_module") as import_module:
            import_module.side_effect = lambda name: {
                "langgraph.checkpoint.postgres": MagicMock(PostgresSaver=fake_saver_cls),
                "langgraph.store.postgres": MagicMock(PostgresStore=fake_store_cls),
            }[name]

            checkpointer, store, pool = main.init_langgraph_persistence("postgresql://test/test")

    create_pool.assert_called_once_with("postgresql://test/test")
    fake_saver_cls.from_conn_string.assert_not_called()
    fake_store_cls.from_conn_string.assert_not_called()
    fake_saver_cls.assert_called_once_with(fake_pool)
    fake_store_cls.assert_called_once_with(fake_pool)
    fake_checkpointer.setup.assert_called_once()
    fake_store.setup.assert_called_once()
    assert checkpointer is fake_checkpointer
    assert store is fake_store
    assert pool is fake_pool


def test_init_langgraph_persistence_closes_pool_on_setup_failure():
    fake_pool = MagicMock(name="pool")
    fake_checkpointer = MagicMock(name="checkpointer")
    fake_checkpointer.setup.side_effect = RuntimeError("setup failed")
    fake_saver_cls = MagicMock(return_value=fake_checkpointer)

    with patch.object(main, "create_langgraph_pg_pool", return_value=fake_pool):
        with patch.object(main.importlib, "import_module") as import_module:
            import_module.side_effect = lambda name: {
                "langgraph.checkpoint.postgres": MagicMock(PostgresSaver=fake_saver_cls),
                "langgraph.store.postgres": MagicMock(PostgresStore=MagicMock()),
            }[name]

            with pytest.raises(RuntimeError, match="setup failed"):
                main.init_langgraph_persistence("postgresql://test/test")

    fake_pool.close.assert_called_once()
