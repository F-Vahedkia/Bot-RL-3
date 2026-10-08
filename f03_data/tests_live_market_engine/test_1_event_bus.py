# f03_data/tests_live_market_engine/test_1_event_bus.py
# Date Reviewed:
#    1405/05/19-18:38 ==> run result is OK for 6 tests

# Run:
# pytest f03_data/tests_live_market_engine/test_1_event_bus.py -v

# اجرای تمام فایلهای تستر داخل پوشه:
# pytest f03_data/tests_live_market_engine/ -v

import pytest
import threading
import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + "/../..")
from f03_data.live_market_engine import EventBus


class TestEventBus:

    # ===================================================== Test-1
    def test_subscribe_and_get_queue(self):
        """
        Test symbol subscription creates queue
        """
        bus = EventBus(queue_size=10)
        symbol = "EURUSD"
        sub_id = bus.subscribe(symbol)
        queue = bus.get_queue(symbol)
        assert queue is not None
        assert queue.maxsize == 10
        assert sub_id == symbol

    # ===================================================== Test-2
    def test_publish_and_get_event(self):
        """
        Test publishing event to symbol subscriber
        """
        bus = EventBus()
        symbol = "EURUSD"
        bus.subscribe(symbol)
        all_dfs = {"EURUSD:H1": "dummy_dataframe"}
        bus.publish(
            event_type="NEW_CANDLE",
            symbol=symbol,
            timeframe="H1",
            all_dfs=all_dfs,
        )
        event = bus.get_event(symbol,timeout=1.0)

        assert event is not None
        assert event["event_type"] == "NEW_CANDLE"
        assert event["symbol"] == "EURUSD"
        assert event["timeframe"] == "H1"
        assert event["all_dfs"] == all_dfs

    # ===================================================== Test-3
    def test_unsubscribe(self):
        """
        Test unsubscribe removes symbol queue
        """
        bus = EventBus()
        symbol = "EURUSD"
        bus.subscribe(symbol)
        assert bus.subscriber_count() == 1
        bus.unsubscribe(symbol)
        assert bus.subscriber_count() == 0
        assert bus.get_queue(symbol) is None

    # ===================================================== Test-4
    def test_queue_full_drops_event(self):
        """
        Test when queue is full, event is dropped
        """
        bus = EventBus(queue_size=1)
        symbol = "EURUSD"
        bus.subscribe(symbol)
        queue = bus.get_queue(symbol)
        queue.put_nowait({"event_type": "FIRST"})

        bus.publish(
            event_type="SECOND",
            symbol=symbol,
            timeframe="H1",
            all_dfs={}
        )
        assert queue.qsize() == 1

    # ===================================================== Test-5
    def test_concurrent_publish_and_subscribe(self):
        """
        Test thread safety
        """
        bus = EventBus()
        symbol = "EURUSD"
        events_received = []


        def consumer():
            bus.subscribe(symbol)
            for _ in range(10):
                event = bus.get_event(symbol, timeout=2.0)
                if event:
                    events_received.append(event)

        def producer():
            # اطمینان از ساخته شدن subscriber
            time.sleep(0.1)
            for i in range(10):
                bus.publish(
                    event_type="NEW_CANDLE",
                    symbol=symbol,
                    timeframe="H1",
                    all_dfs={"counter": i}
                )
                time.sleep(0.01)

        consumer_thread = threading.Thread(target=consumer)
        producer_thread = threading.Thread(target=producer)
        consumer_thread.start()
        producer_thread.start()

        producer_thread.join()
        consumer_thread.join(timeout=3)
        assert len(events_received) == 10

    # ===================================================== Test-6
    def test_stats(self):
        """
        Test stats method
        """
        bus = EventBus()
        symbol1 = "EURUSD"
        symbol2 = "GBPUSD"
        bus.subscribe(symbol1)
        bus.subscribe(symbol2)
        bus.publish(
            event_type="NEW_CANDLE",
            symbol=symbol1,
            timeframe="H1",
            all_dfs={}
        )
        bus.publish(
            event_type="NEW_CANDLE",
            symbol=symbol2,
            timeframe="H1",
            all_dfs={}
        )
        stats = bus.stats()

        assert symbol1 in stats
        assert symbol2 in stats

        assert stats[symbol1] == 1
        assert stats[symbol2] == 1

# ============================================================================= END