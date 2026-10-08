

    def download_historical_data_old1(self,
                                  symbols: Optional[List[str]] = None,
                                  timeframes: Optional[List[str]] = None,
                                  lookback_bars: Optional [int] = None) -> None:
        if not self.downloader:
            self.logger.warning("Downloader not initialized.")
            return
        self.logger.info("Starting historical data download...")
        
        if lookback_bars is None:
            lookback_bars = self.cfg.get("download_defaults", {}).get("lookback_bars", 1000)

        try:
            if not self.downloader.conn.initialize():
                raise RuntimeError("MT5 connection failed")
            plans = self.downloader.build_plan(
                symbols=symbols,
                timeframes=timeframes,
                lookback_bars=lookback_bars
            )
            results = self.downloader.run(plans)
            for r in results:
                if "error" in r:
                    self.logger.error(f"Failed: {r['symbol']}/{r['timeframe']} - {r['error']}")
                else:
                    self.logger.info(f"Downloaded: {r['symbol']}/{r['timeframe']} - {r['rows_written']} rows")
            self.logger.info("Historical data download completed")
        except Exception as e:
            self.logger.exception(f"Download failed: {e}")
        finally:
            self.downloader.conn.shutdown()

    def download_historical_data_old2(self,
                                  symbols: Optional[List[str]] = None,
                                  timeframes: Optional[List[str]] = None,
                                  lookback_bars: Optional[int] = None) -> None:
        if not self.downloader:
            self.logger.warning("Downloader not initialized.")
            return
        
        # خواندن از کانفیگ اگر در خط فرمان داده نشده باشد
        dl_cfg = self.cfg.get("download_defaults", {})
        if not symbols:
            symbols = dl_cfg.get("symbols", [])
        if not timeframes:
            timeframes = dl_cfg.get("timeframes", [])
        if lookback_bars is None:
            lookback_bars = dl_cfg.get("lookback_bars", 1000)
        
        self.logger.info("Starting historical data download...")
        self.logger.info(f"Parameters: symbols={symbols}, timeframes={timeframes}, lookback_bars={lookback_bars}")
        
        try:
            if not self.downloader.conn.initialize():
                raise RuntimeError("MT5 connection failed")
            
            # ساخت پلن با استفاده از کانفیگ
            plans = self.downloader.build_plan(
                symbols=symbols,
                timeframes=timeframes,
                lookback_bars=lookback_bars,
                date_from=dl_cfg.get("date_from"),
                date_to=dl_cfg.get("date_to")
            )
            
            results = self.downloader.run(plans)
            for r in results:
                if "error" in r:
                    self.logger.error(f"Failed: {r['symbol']}/{r['timeframe']} - {r['error']}")
                else:
                    self.logger.info(f"Downloaded: {r['symbol']}/{r['timeframe']} - {r['rows_written']} rows")
            self.logger.info("Historical data download completed")
        except Exception as e:
            self.logger.exception(f"Download failed: {e}")
        finally:
            self.downloader.conn.shutdown()

    def download_historical_data_old3(self,
                                  symbols: Optional[List[str]] = None,
                                  timeframes: Optional[List[str]] = None,
                                  lookback_bars: Optional[int] = None) -> None:
        if not self.downloader:
            self.logger.warning("Downloader not initialized.")
            return
        
        dl_cfg = self.cfg.get("download_defaults", {})
        if not symbols:
            symbols = dl_cfg.get("symbols", [])
        if not timeframes:
            timeframes = dl_cfg.get("timeframes", [])
        if lookback_bars is None:
            lookback_bars = dl_cfg.get("lookback_bars", 1000)
        
        self.logger.info("Starting historical data download...")
        self.logger.info(f"Parameters: symbols={symbols}, timeframes={timeframes}, lookback_bars={lookback_bars}")
        
        try:
            if not self.downloader.conn.initialize():
                raise RuntimeError("MT5 connection failed")
            
            # تبدیل تاریخ به فرمت ISO
            date_from = dl_cfg.get("date_from")
            if date_from and isinstance(date_from, str) and len(date_from) == 10:
                date_from = f"{date_from}T00:00:00Z"
            
            date_to = dl_cfg.get("date_to")
            if date_to and isinstance(date_to, str) and date_to.lower() == "now":
                date_to = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
            plans = self.downloader.build_plan(
                symbols=symbols,
                timeframes=timeframes,
                lookback_bars=lookback_bars,
                date_from=date_from,
                date_to=date_to
            )
            
            results = self.downloader.run(plans)
            for r in results:
                if "error" in r:
                    self.logger.error(f"Failed: {r['symbol']}/{r['timeframe']} - {r['error']}")
                else:
                    self.logger.info(f"Downloaded: {r['symbol']}/{r['timeframe']} - {r['rows_written']} rows")
            self.logger.info("Historical data download completed")
        except Exception as e:
            self.logger.exception(f"Download failed: {e}")
        finally:
            self.downloader.conn.shutdown()

