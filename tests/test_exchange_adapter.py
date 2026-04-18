"""Tests para exchange_adapter.py -- Multi-exchange abstraction."""

import unittest
from unittest.mock import patch, MagicMock

import exchange_adapter


class TestExchangeAdapterBase(unittest.TestCase):
    def test_base_raises_not_implemented(self):
        adapter = exchange_adapter.ExchangeAdapter()
        with self.assertRaises(NotImplementedError):
            adapter.get_price("BTCUSDT")
        with self.assertRaises(NotImplementedError):
            adapter.get_candles("BTCUSDT")
        with self.assertRaises(NotImplementedError):
            adapter.place_order("BTCUSDT", "BUY", 0.001)
        with self.assertRaises(NotImplementedError):
            adapter.ping()


class TestBinanceAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = exchange_adapter.BinanceAdapter()

    def test_mainnet_url(self):
        a = exchange_adapter.BinanceAdapter(testnet=False)
        self.assertIn("api.binance.com", a.base_url)

    def test_testnet_url(self):
        a = exchange_adapter.BinanceAdapter(testnet=True)
        self.assertIn("testnet", a.base_url)

    @patch("exchange_adapter.requests.get")
    def test_get_price(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"symbol": "BTCUSDT", "price": "50000.00"},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        price = self.adapter.get_price("BTCUSDT")
        self.assertEqual(price, 50000.0)

    @patch("exchange_adapter.requests.get")
    def test_get_price_error(self, mock_get):
        mock_get.side_effect = Exception("network error")
        price = self.adapter.get_price("BTCUSDT")
        self.assertIsNone(price)

    @patch("exchange_adapter.requests.get")
    def test_get_candles(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [[1000, "50000", "51000", "49000", "50500", "100"]],
        )
        mock_get.return_value.raise_for_status = MagicMock()
        candles = self.adapter.get_candles("BTCUSDT")
        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0][4], 50500.0)  # close

    @patch("exchange_adapter.requests.get")
    def test_get_orderbook(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"bids": [["50000", "1.5"]], "asks": [["50001", "0.5"]]},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        ob = self.adapter.get_orderbook("BTCUSDT")
        self.assertEqual(len(ob["bids"]), 1)
        self.assertEqual(ob["bids"][0][0], 50000.0)

    @patch("exchange_adapter.requests.get")
    def test_ping_ok(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {})
        mock_get.return_value.raise_for_status = MagicMock()
        self.assertTrue(self.adapter.ping())

    @patch("exchange_adapter.requests.get")
    def test_ping_fail(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        self.assertFalse(self.adapter.ping())

    def test_signed_request_no_keys(self):
        with self.assertRaises(ValueError):
            self.adapter._signed_request("GET", "/api/v3/account")

    @patch("exchange_adapter.requests.get")
    def test_symbol_info(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "filters": [
                {"filterType": "LOT_SIZE", "minQty": "0.00001", "stepSize": "0.00001"},
                {"filterType": "NOTIONAL", "minNotional": "10"},
            ]}]},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        info = self.adapter.get_symbol_info("BTCUSDT")
        self.assertEqual(info["min_qty"], 0.00001)
        self.assertEqual(info["min_notional"], 10.0)


class TestBybitAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = exchange_adapter.BybitAdapter()

    @patch("exchange_adapter.requests.get")
    def test_get_price(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": {"list": [{"lastPrice": "3000.50"}]}},
        )
        price = self.adapter.get_price("ETHUSDT")
        self.assertEqual(price, 3000.5)

    @patch("exchange_adapter.requests.get")
    def test_ping(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        self.assertTrue(self.adapter.ping())

    def test_place_order_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.adapter.place_order("BTCUSDT", "BUY", 0.01)


class TestFactory(unittest.TestCase):
    def test_create_binance(self):
        a = exchange_adapter.create_adapter("binance")
        self.assertIsInstance(a, exchange_adapter.BinanceAdapter)

    def test_create_bybit(self):
        a = exchange_adapter.create_adapter("bybit")
        self.assertIsInstance(a, exchange_adapter.BybitAdapter)

    def test_create_unknown_raises(self):
        with self.assertRaises(ValueError):
            exchange_adapter.create_adapter("kraken")

    def test_list_exchanges(self):
        exchanges = exchange_adapter.list_exchanges()
        self.assertIn("binance", exchanges)
        self.assertIn("bybit", exchanges)


if __name__ == "__main__":
    unittest.main()
