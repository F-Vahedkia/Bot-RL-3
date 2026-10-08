# f04_features/feature_C_engine_6_tester_A.py
# Date reviewed:
#    1405/05/24-15:06 ==> run result is OK for 12 tests

# Run: pytest -v -s f04_features/feature_C_engine_6_tester_A.py


from __future__ import annotations
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import f04_features.feature_C_engine_6 as em
from f03_data.mtf_dataset import MTFDataset
from f04_features.feature_C_engine_6 import FeatureEngine


def frame(values=(1.0, 2.0, 3.0)):
    idx = pd.date_range('2026-01-01 10:00:00', periods=len(values), freq='min', tz='UTC', name='timestamp')
    return pd.DataFrame({'open': values, 'high':[v+.5 for v in values], 'low':[v-.5 for v in values], 'close':values, 'volume':range(100,100+len(values))}, index=idx)


def dataset(symbol='XAUUSD'):
    ds = MTFDataset(symbol=symbol, base_tf='M1')
    ds.add('M1', frame())
    return ds


def ps(name='sma', raw='sma(column="close",period=2)@M1', canonical=None, kwargs=None):
    canonical = canonical or raw
    return SimpleNamespace(raw=raw, canonical=canonical, name=name, timeframe='M1', args=(), kwargs=kwargs or {'column':'close','period':2})


def spec(name='sma', fn=None, stateful=False, output_names=None):
    modes = {'live','paper','shadow','backtest','replay','eval'} if stateful else {'train','optimize'}
    return SimpleNamespace(name=name, fn=fn or (lambda *a, **k: None), is_stateful=stateful, modes=modes,
                           required_cols=['close'], output_names=output_names,
                           supports=lambda m: m in modes,
                           is_batch_mode=lambda m: m in {'train','optimize'},
                           is_incremental_mode=lambda m: m in modes-{'train','optimize'})


class Live:
    def __init__(self, **kwargs): self.kwargs = kwargs
    def update(self, x): return x * 10


def test_init_validation():
    with pytest.raises(ValueError, match='config is required'): FeatureEngine(None)
    with pytest.raises(TypeError, match='config must be dict'): FeatureEngine([])
    e = FeatureEngine({})
    assert e.mode is None and e._run_id == 0 and e._live_cache == {}


def test_reset_live_state():
    e = FeatureEngine({}); e.mode='live'; e._live_cache['x']={}
    e.reset_live_state()
    assert e.mode is None and e._live_cache == {}


def test_execute_validation():
    e = FeatureEngine({}); ds=dataset()
    with pytest.raises(ValueError, match='dataset is required'): e.execute(None, [])
    with pytest.raises(TypeError, match='Expected MTFDataset'): e.execute('bad', [])
    assert e.execute(MTFDataset(symbol='XAUUSD', base_tf='M1'), []) is not None
    with pytest.raises(ValueError, match='specs is required'): e.execute(ds, None)
    assert e.execute(ds, []) is ds


def test_execute_context_and_parse(monkeypatch):
    e=FeatureEngine({}); ds=dataset()
    monkeypatch.setattr(em, 'parse_spec', lambda spec, mode: ps())
    monkeypatch.setattr(em, 'get_indicator', lambda name, mode: spec())
    monkeypatch.setattr(e, '_validate_contract', lambda **k: True)
    monkeypatch.setattr(e, '_call_batch', lambda spec, dataset, ps: dataset)
    monkeypatch.setattr(em, 'cached_compute', lambda **k: k['compute_fn']())
    assert e.execute(ds, ['x'], mode='train') is ds
    assert e.mode == 'train' and e._run_id == 1


def test_execute_all_invalid_specs(monkeypatch):
    e=FeatureEngine({})
    monkeypatch.setattr(em, 'parse_spec', lambda *a, **k: (_ for _ in ()).throw(ValueError('bad')))
    with pytest.raises(ValueError, match='No valid feature specifications were parsed'):
        e.execute(dataset(), ['bad'])


def test_validate_contract():
    e=FeatureEngine({}); s=spec()
    assert e._validate_contract(s, frame(), 'train', 'M1')
    assert not e._validate_contract(s, frame(), 'live', 'M1')
    assert not e._validate_contract(s, frame().drop(columns=['close']), 'train', 'M1')
    f=frame().drop(columns=['close']); f['M1_close']=[1,2,3]
    assert e._validate_contract(s, f, 'train', 'M1')


def test_merge():
    e=FeatureEngine({}); left=frame(); right=pd.DataFrame({'x':[1,2,3]}, index=left.index)
    out=e._merge(left,right); assert 'x' in out and 'x' not in left
    assert e._merge(left, pd.DataFrame({'close':left.close.copy()}, index=left.index)).close.equals(left.close)
    with pytest.raises(ValueError, match='Feature column collision'):
        e._merge(left, pd.DataFrame({'close':[9,9,9]}, index=left.index))
    assert e._merge(left, pd.DataFrame(index=left.index)) is left


def test_normalize_output():
    e=FeatureEngine({})
    s=spec('macd', stateful=True, output_names=['macd','macd_signal','macd_hist'])
    assert e._normalize_output(s,(1,2,3)) == {'macd':1,'macd_signal':2,'macd_hist':3}
    assert e._normalize_output(s,{'macd':1}) == {'macd':1}
    with pytest.raises(TypeError, match='returned scalar output'): e._normalize_output(s,5)
    s1=spec('sma', stateful=True, output_names=['sma']); assert e._normalize_output(s1,5)=={'sma':5}
    s0=spec('sma', stateful=True); assert e._normalize_output(s0,5)=={'sma':5}


def test_live_helpers():
    e=FeatureEngine({}); s=spec('sma', fn=lambda **k: Live(**k), stateful=True, output_names=['sma'])
    obj=e._build_live_instance(s, ps()); assert isinstance(obj, Live)
    assert e._update_live(obj, frame().iloc[0], s, ps()) == {'sma':10}
    assert e._update_live(obj, pd.Series({'open':1.0}), s, ps()) is None
    assert e._build_live_column_name('sma', ps()) == ps().canonical
    p=ps('macd', raw='macd(fast=12,slow=26,signal=9)@M1', canonical='macd(fast=12,slow=26,signal=9)@M1')
    assert e._build_live_column_name('macd_signal', p) == 'macd_signal(fast=12,slow=26,signal=9)@M1'


def test_call_batch_single_and_multi_output():
    e=FeatureEngine({})
    def one(df, **k): return pd.DataFrame({'v':[7,8,9]}, index=df.index)
    s=spec('sma', fn=one, output_names=['sma']); ds=dataset(); e._call_batch(s,ds,ps()); assert ps().canonical in ds.get('M1')
    def multi(df, **k): return pd.DataFrame({'a':[1,2,3],'b':[4,5,6]}, index=df.index)
    s=spec('macd', fn=multi, output_names=['macd','macd_signal'])
    p=ps('macd', raw='macd(fast=12,slow=26,signal=9)@M1', canonical='macd(fast=12,slow=26,signal=9)@M1', kwargs={'fast':12,'slow':26,'signal':9})
    ds=dataset(); e._call_batch(s,ds,p); cols=ds.get('M1').columns
    assert 'macd(fast=12,slow=26,signal=9)@M1' in cols and 'macd_signal(fast=12,slow=26,signal=9)@M1' in cols


def test_call_batch_errors():
    e=FeatureEngine({})
    def two(df, **k): return pd.DataFrame({'a':[1,2,3],'b':[4,5,6]}, index=df.index)
    with pytest.raises(ValueError, match='Output count mismatch'):
        e._call_batch(spec('macd', fn=two, output_names=['macd']), dataset(), ps('macd'))
    with pytest.raises(TypeError, match='must return DataFrame'):
        e._call_batch(spec('sma', fn=lambda *a,**k:[1,2,3], output_names=['sma']), dataset(), ps())


def test_attach_live_output():
    e=FeatureEngine({}); df=frame(); p=ps(); s=spec('sma', stateful=True, output_names=['sma'])
    out=e._attach_live_output(df,p,s,{df.index[0]:{'sma':10.0}, df.index[1]:{'sma':20.0}})
    assert p.canonical in out.columns and out.loc[df.index[0],p.canonical]==10
    assert e._attach_live_output(df,p,s,{}) is df
