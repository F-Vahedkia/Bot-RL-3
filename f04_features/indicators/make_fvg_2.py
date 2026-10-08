
def make_fvg(df: pd.DataFrame, **cfg) -> Dict[str, pd.Series]:
    """
    Used from function: detect_fvg_optimized()
    Factory رجیستری: دریافت DataFrame با ستون‌های 'open','high','low','close'
    و برگرداندن خروجی استاندارد FVG (برای Engine/Registry).

    cfg شامل پارامترهای detect_fvg است:
      - lookback: int
      - atr_window: int
      - min_size_pct_of_atr: float
    """
    # --- Used in Step-3 ------------------------
    # سطر زیر بیان میکند که هر اِف وی جی چند کندل فرصت دارد تا لمس شود
    N = int(cfg.get("max_bars_alive", 4))
    # --- sanity check ---
    if N < 1:
        raise ValueError(f"max_bars_alive must be >= 1, got {N}")
    if N > 50:
        # یا عددی که برای سیستم خودت مناسب می‌دانی
        raise ValueError(f"max_bars_alive too large for production: {N}")

    # --- Used in Step-4 (subpart-5) ------------
    w_size  = float(cfg.get("w_size" , 0.5))
    w_age   = float(cfg.get("w_age"  , 0.3))
    w_touch = float(cfg.get("w_touch", 0.2))
    # --- sanity check ---
    if any(w < 0 for w in (w_size, w_age, w_touch)):
        raise ValueError("FVG weights (w_size, w_age, w_touch) must be non-negative.")

    w_sum = w_size + w_age + w_touch
    if w_sum == 0:
        raise ValueError("Sum of FVG weights must be > 0.")
    # نرمال‌سازی
    w_size  /= w_sum
    w_age   /= w_sum
    w_touch /= w_sum

    # --- اعتبارسنجی ورودی ------------------------------------------ ok
    needed = ("open", "high", "low", "close")
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
    
    # === Step-1: Basic Output of FVG ========================================= ok
    out = detect_fvg_optimized(
        df["open"], df["high"], df["low"], df["close"],
        lookback=int(cfg.get("lookback", 2)),
        atr_window=int(cfg.get("atr_window", 14)),
        min_size_pct_of_atr=float(cfg.get("min_size_pct_of_atr", 0.50)),
        use_middle_filter=True,
    )
    # assert output structure
    for c in ("fvg_top", "fvg_bottom", "fvg_up", "fvg_down"):
        if c not in out:
            raise KeyError(f"detect_fvg_optimized missing '{c}'")
    if not out.index.equals(df.index):
        out = out.reindex(df.index)
        
    # === Step-2: One-Bar Lifecycle (without look-ahead) ====================== ok
    # نکته: یک شیفت 1 کندلی در تمام خروجی های توابع تشخیص اِف وی جی قبلاً اعمال شده است

    # یافتن کف و سقف زون اِف وی جی
    up  = out["fvg_up"].astype(bool)
    dn  = out["fvg_down"].astype(bool)
    top = out["fvg_top"]
    bot = out["fvg_bottom"]

    # تشخیص اینکه اصلاً زونی در کندل قبلی وجود دارد یا نه
    # این زون در صورت وجود در کندل جاری ذخیره شده است
    prev_exists = top.notna() & bot.notna()
    # تشخیص تولد زون: زون در کندل قبلی وجود دارد ولی در دو کندل قبلی وجود نداشته است
    born = prev_exists & ~(prev_exists.shift(1).astype(bool).fillna(False))

    # The union of the two sets "filled" and "expired" is equal to the set "touched".
    # The intersection of the two sets "filled" and "expired" is the empty set.
    touched_next = ((        # دو سطر زیر لمس زون توسط کندل 4 را بررسی میکنند
        ((df["low" ] <= top) & up ) |  # این سطر چک میکند که در حالت صعودی کف کندل 4 به زیر سقف زون آمده
        ((df["high"] >= bot) & dn )    # این سطر چک میکند که در حالت نزولی سقف کندل 4 به بالای کف زون آمده
    ) &
        prev_exists          # این سطر بررسی میکند که آیا در کندل 4 زون اِف وی جی مربوط به کندلهای 1و2و3 ثبت شده است؟
    )

    filled_next = ((         # دو سطر زیر پوشش کامل زون توسط کندل 4 را بررسی میکنند
        ((df["low" ] <= bot) & up ) | # این سطر چک میکند که در حالت صعودی کف کندل 4 به کف زون رسیده
        ((df["high"] >= top) & dn )   # این سطر چک میکند که در حالت نزولی سقف کندل 4 به سقف زون رسیده
    ) &
        prev_exists          # این سطر بررسی میکند که آیا در کندل 4 زون اِف وی جی مربوط به کندلهای 1و2و3 ثبت شده است؟
    )
    
    expired_next = (
        prev_exists &        # این سطر بررسی میکند که آیا در کندل 4 زون اِف وی جی مربوط به کندلهای 1و2و3 ثبت شده است؟
        touched_next &       # 
        (~filled_next)       # این سطر بررسی میکند که آیا زون کندلهای 1و2و3 در این کندل 4 پُر نشده است؟
    )
    
    out.update({             # قبل از خروجی نهایی، مقادیر درست/نادرست به مقادیر 1/0 تبدیل میشوند
        "fvg_born"        :         born.astype("int8"),
        "fvg_touched_next": touched_next.astype("int8"), 
        "fvg_filled_next" :  filled_next.astype("int8"),
        "fvg_expired_next": expired_next.astype("int8"),
    })

    # === Step-3: Multi-Bar Lifecycle ========================================= ok
    # جمع آوری زونهای اِن کندل گذشته
    tops = [out["fvg_top"]   .shift(k) for k in range(N)]  # = range(0,N)
    bots = [out["fvg_bottom"].shift(k) for k in range(N)]
    ups  = [out["fvg_up"]    .shift(k) for k in range(N)]
    dns  = [out["fvg_down"]  .shift(k) for k in range(N)]

    # وجود زون در پنجرهٔ 0...اِن-1 -----------------------------------
    # بررسی این که آیا در این پنجره زمانی، زونی وجود داشته است یا نه؟
    """
    در زیر یک لیست اِن تایی از سری ها داریم سری اول دارای شیفت 1 کندلی، سری دوم دارای شیفت 2 کندلی و ... هستند
    هر سری دارای مقادیر درست/نادرست است
    مثلاً سومین سری دارای مقادیر زیر است:
    out["fvg_top"].shift(3).notna & out["fvg_bottom"].shift(3).notna
    """
    has_zone_cols = [(t.notna() & b.notna()) for t, b in zip(tops, bots)]
    # این خط می‌گوید: «آیا در اِن کندل قبلی، در هیچ‌کدام از آن‌ها زونی وجود داشته؟»
    # has_zone = pd.concat(has_zone_cols, axis=1).any(axis=1) # هزینه محاسباتی زیاد است. با سطرهای زیر جایگزین میشود
    has_zone = np.column_stack([c.to_numpy() for c in has_zone_cols]).any(axis=1)  # سطر کم هزینه و جایگزین
    has_zone = pd.Series(has_zone, index=df.index)                                 # سطر کم هزینه و جایگزین


    # تاچ کندل اخیر با هر کدام از زون‌های پنجرهٔ 0...اِن-1 ----------
    touched_list = [((df["low"] <= t) & (df["high"] >= b)) for t, b in zip(tops, bots)]
    # touched_any = pd.concat(touched_list, axis=1).any(axis=1)  # هزینه محاسباتی زیاد است. با سطرهای زیر جایگزین میشود
    touched_any = np.column_stack([c.to_numpy() for c in touched_list]).any(axis=1)  # سطر کم هزینه و جایگزین
    touched_any = pd.Series(touched_any, index=df.index)                             # سطر کم هزینه و جایگزین

    # پُرشدن هر کدام از زون‌های پنجرهٔ 0...اِن-1 با کندل اخیر -------
    filled_list_up = [((u.astype(bool)) & (df["low"] <= b)) for u, b in zip(ups, bots)]
    filled_list_dn = [((d.astype(bool)) & (df["high"] >= t)) for d, t in zip(dns, tops)]


    # filled_any = (                                        # هزینه محاسباتی زیاد است. با سطرهای زیر جایگزین میشود
    #     pd.concat(filled_list_up, axis=1).any(axis=1) |
    #     pd.concat(filled_list_dn, axis=1).any(axis=1)     )
    filled_up_any = np.column_stack([c.to_numpy() for c in filled_list_up]).any(axis=1)  # سطر کم هزینه و جایگزین
    filled_dn_any = np.column_stack([c.to_numpy() for c in filled_list_dn]).any(axis=1)  # سطر کم هزینه و جایگزین
    filled_any = filled_up_any | filled_dn_any                                           # سطر کم هزینه و جایگزین
    filled_any = pd.Series(filled_any, index=df.index)                                   # سطر کم هزینه و جایگزین

    # --- Expire (فقط قدیمی‌ترین زون) -------------------------------
    t_old, b_old = tops[-1], bots[-1]
    touch_old = t_old.notna() & b_old.notna() & (df["low"] <= t_old) & (df["high"] >= b_old)
    expired_now = t_old.notna() & b_old.notna() & (~touch_old)

    # --- Alive (حداقل یک زون در پنجره باشد که پُـر نشده باشد) ----
    # --- Preparing input data ------------------
    high_ = df["high"].to_numpy()
    low_  = df["low"].to_numpy()
    tops_mat = np.column_stack([t.to_numpy() for t in tops])
    bots_mat = np.column_stack([b.to_numpy() for b in bots])
    ups_mat  = np.column_stack([u.to_numpy() for u in ups])
    dns_mat  = np.column_stack([d.to_numpy() for d in dns])

    # --- نسخه اولیه ----------------------------
    def compute_alive_orig(low, high, tops, bots, ups, dns):
        alive_list = []
        for t, b, u, d in zip(tops, bots, ups, dns):
            zone_exists = t.notna() & b.notna()
            zone_filled = (
                ((low  <= b) & u) |
                ((high >= t) & d)
            ) & zone_exists
            zone_alive = zone_exists & (~zone_filled)
            alive_list.append(zone_alive)

        alive_now = pd.Series(
            np.column_stack([a.to_numpy() for a in alive_list]).any(axis=1),
            index=low.index
        )
        return alive_now
    
    # --- نسخه سریعتر ---------------------------
    def compute_alive_numpy(low, high, tops_, bots_, ups_, dns_):
        low_  = low.to_numpy()[:, None]
        high_ = high.to_numpy()[:, None]
        zone_exists = ~np.isnan(tops_) & ~np.isnan(bots_)
        zone_filled = (
            (ups_ & (low_  <= bots_)) |
            (dns_ & (high_ >= tops_))
        )
        zone_alive = zone_exists & (~zone_filled)
        alive_now = pd.Series(zone_alive.any(axis=1), index=low.index)
        return alive_now

    # --- نسخه فوق سریع -------------------------
    @njit
    def compute_alive_ultra(low_, high_, tops_, bots_, ups_, dns_):
        T, N = tops_.shape
        alive = np.zeros(T, dtype=np.bool_)
        for i in range(T):
            alive_flag = False
            for j in range(N):
                t = tops_[i, j]
                b = bots_[i, j]
                if np.isnan(t) or np.isnan(b):
                    continue

                filled = False
                if ups_[i, j]:
                    if low_[i] <= b:
                        filled = True
                elif dns_[i, j]:
                    if high_[i] >= t:
                        filled = True

                if not filled:
                    alive_flag = True
                    break

            alive[i] = alive_flag
        return alive

    # --- Alive result --------------------------
    # alive_now = compute_alive_orig(df["low"], df["high"], tops, bots, ups, dns)
    # alive_now = compute_alive_numpy(df["low"], df["high"], tops_mat, bots_mat, ups_mat, dns_mat)
    alive_now = pd.Series(
        compute_alive_ultra(low_, high_, tops_mat, bots_mat, ups_mat, dns_mat),
        index = df.index
    )

    out.update({
        "fvg_touched_any": touched_any.astype("int8"),   # تعیین میکند که آیا هیچکدام از زونهای داخل پنجره لمس شده اند
        "fvg_filled_any":   filled_any.astype("int8"),   # تعیین میکند که آیا هیچکدام از زونهای داخل پنجره پـُر شده اند
        "fvg_expired_now": expired_now.astype("int8"),   # تعیین میکند که آیا قدیمی ترین زون منقضی شده است
        "fvg_alive_window":  alive_now.astype("int8"),   # تعیین میکند که آیا حداقل یک زون در پنجره هست که پـُر نشده باشد
    })

    # === Step‑4: Propagate Zone to Engine ====================================
    # Propagate the zone onto the target timeframe for easy consumption in the Engine

    # Selectin the newest/nearest active zone in range 0..N-1 ------- subpart-1
    # ستون 0: shift(0),..., ستون آخر: shift(N-1)
    # cand_top = pd.concat(tops, axis=1).astype("float32")  # هزینه محاسباتی زیاد است. با سطرهای زیر جایگزین میشود
    # cand_bot = pd.concat(bots, axis=1).astype("float32")  # هزینه محاسباتی زیاد است. با سطرهای زیر جایگزین میشود
    
    cand_top = pd.DataFrame(                                # سطر کم هزینه و جایگزین
        np.column_stack([s.to_numpy() for s in tops]),
        index=df.index,
        dtype="float32"
    )

    cand_bot = pd.DataFrame(                                # سطر کم هزینه و جایگزین
        np.column_stack([s.to_numpy() for s in bots]),
        index=df.index,
        dtype="float32"
    )

    # انتخاب اولین مقدار غیر نن (معتبر) از چپ -------------
    sel_top = cand_top.bfill(axis=1).iloc[:, 0]
    sel_bot = cand_bot.bfill(axis=1).iloc[:, 0]
    # انتخاب فقط زونهایی که واقعاً فعال هستند -------------
    active_top = sel_top.where(alive_now)
    active_bot = sel_bot.where(alive_now)


    # --- Zone size & Normalization by ATR -------------------------- subpart-2
    atrv = atr_core(df["high"], df["low"], df["close"], n=int(cfg.get("atr_window", 14))).astype("float32")
    gap_size = (active_top - active_bot).abs().astype("float32")
    size_norm = (gap_size / (atrv.clip(lower=1e-8))).astype("float32")

    # Calc. zone age (dist from newest selected zone) --------------- subpart-3
    # در بخش قبل، cand_top و sel_top ساخته شده‌اند
    # سن زون برابر است با شماره ستونی که sel_top از آن آمده (0..N-1)
    
    # cand_top = cand_top.copy()  # چون تاانتهای این تابع دیگراز این دیتافریم استفاده نمیشود،این سطررا کامنت میکنم
    cand_top.columns = np.arange(cand_top.shape[1])  # changing column names to 0,1,...,N-1
    cmp = cand_top.eq(sel_top, axis=0).to_numpy()  # in which column of cand_top, the value is equal to sel_top
                                                   # cmp will be a DataFrame contains TRUE/FALSE
                                                   # در هر ردیف، ستونی که (درست) است همان ستونی است که زون انتخاب شده از آن آمده
    any_true = cmp.any(axis=1)  # تعیین میکند که هر ردیف اصلاً زون معتبری داشت یا نه
                                # any_true: will be a one-dimensional vector
    """ for below line:
    for i in rows(any_ture):
        if any_true[i] == True:
            idx[i] = argmax_row[i] + 1
        else:
            idx[i] = NaN
    """
    idx = np.where(any_true, cmp.argmax(axis=1) + 1, np.nan)  # اگر هیچ True نبود → NaN
    age_bars_est = pd.Series(idx, index=df.index, dtype="float32").where(active_top.notna())


    # --- Current touch & counting touches in window=N -------------- subpart-4
    touch_now = (
        (df["low"] <= active_top) & (df["high"] >= active_bot) &
        active_top.notna() & active_bot.notna()
    ).astype("int8")
    # به‌عنوان تقریب عملیاتی: مجموع لمس‌ها در پنجرهٔ ثابت N (برای زون فعال)
    touch_count = (
        touch_now
        .rolling(N, min_periods=1).sum().where(active_top.notna()).fillna(0).astype("float32")
    )

    
    # --- امتیاز زون: ترکیب اندازه/قدمت/تعداد لمس (بدون look-ahead) --- subpart-5

    # Simple Normilization of age/touch -----------------------------
    age_score = 1.0 / (1.0 + age_bars_est)           # هر چقدر که زون جدیدتر باشد امتیازش بیشتر است
    touch_score = (touch_count / float(max(1, N)))   # لمس‌های بیشتر (در N) → امتیاز بالاتر
    # Calc. fvg_score by weighted mean-------------------------------
    score = (w_size  *   size_norm.fillna(0.0) +   # size_norm  : comes form subpart-2
             w_age   *   age_score.fillna(0.0) +   # age_score  : comes form subpart-5
             w_touch * touch_score.fillna(0.0)     # touch_score: comes form subpart-5
    ).astype("float32")

    out.update({
        "fvg_active_top":    active_top.astype("float32"),      # from subpart-1
        "fvg_active_bottom": active_bot.astype("float32"),      # from subpart-1
        "fvg_gap_size":           gap_size.fillna(0).astype("float32"),   # from subpart-2
        "fvg_size_norm":         size_norm.fillna(0).astype("float32"),   # from subpart-2
        "fvg_age_bars_est":   age_bars_est.fillna(0).astype("float32"),        # from subpart-3
        "fvg_touch_now":      touch_now.astype("int8"),       # from subpart-4
        "fvg_touch_count":  touch_count.astype("float32"),    # from subpart-4
        "fvg_score":              score.astype("float32"),         # from subpart-5
    })
    return out
