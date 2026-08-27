import io, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import psychrolib

GHCN_ID='USW00004880'
ISD_ID='72212704880'
STATION_NAME='GREATER KANKAKEE AIRPORT'
ELEVATION_M=191.7
LOCAL_TZ='America/Chicago'
START_LOCAL=pd.Timestamp('2011-08-26 00:00:00',tz=LOCAL_TZ)
END_LOCAL=pd.Timestamp('2026-08-26 23:00:00',tz=LOCAL_TZ)
START_UTC=START_LOCAL.tz_convert('UTC')
END_UTC=END_LOCAL.tz_convert('UTC')
TARGET=pd.date_range(START_UTC,END_UTC,freq='h')
OUT=Path('Kankakee_Kensing_Hourly_WetBulb_Weather_2011-08-26_to_2026-08-26.csv')
QA=Path('Kankakee_Kensing_Hourly_WetBulb_QA.json')
S=requests.Session(); S.headers.update({'User-Agent':'Mozilla/5.0 engineering-weather-analysis/1.0','Accept':'text/csv,text/plain,*/*'})

USER_CANDIDATES=[
'https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/2024/USW00004880.csv',
'https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/2024/LCD_USW00004880_2024.csv',
'https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/by-year/2024/LCD_USW00004880_2024.csv',
'https://www.ncei.noaa.gov/data/local-climatological-data/access/2024/USW00004880.csv',
'https://www.ncei.noaa.gov/data/local-climatological-data-v2/access/2024/LCD_USW00004880_2024.csv',
]

def req(url,timeout=60):
    last=None
    for n in range(3):
        try:
            r=S.get(url,timeout=timeout,allow_redirects=True); last=r
            if r.status_code==200 and r.content: return r
        except Exception as e: last=e
        time.sleep(1+n)
    if isinstance(last,Exception): raise last
    return last

def candidate_test():
    out=[]
    for u in USER_CANDIDATES:
        try:
            r=S.get(u,timeout=8,allow_redirects=True)
            out.append({'url':u,'status':r.status_code,'bytes':len(r.content),'content_type':r.headers.get('content-type','')})
        except Exception as e: out.append({'url':u,'status':'EXCEPTION','detail':repr(e)})
    return out

def isd_temp(s):
    raw=s.astype('string').str.split(',',n=1).str[0]
    v=pd.to_numeric(raw,errors='coerce')/10.0
    missing=raw.str.replace('+','',regex=False).str.replace('-','',regex=False).eq('9999')
    return v.mask(missing)

def fetch_noaa_year(year):
    url=f'https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{ISD_ID}.csv'
    r=req(url,90)
    if r.status_code!=200: raise RuntimeError(f'{url} status={r.status_code}')
    d=pd.read_csv(io.BytesIO(r.content),dtype=str,low_memory=False)
    if not {'DATE','TMP','DEW'}.issubset(d.columns): raise RuntimeError(f'Bad NOAA columns {year}: {d.columns.tolist()[:20]}')
    z=pd.DataFrame({'obs_ts':pd.to_datetime(d.DATE,errors='coerce',utc=True),'temp_c':isd_temp(d.TMP),'dew_c':isd_temp(d.DEW)})
    z['source']='NOAA_NCEI_GLOBAL_HOURLY'; z['source_year']=year
    return z.dropna(subset=['obs_ts']), {'year':year,'source':'NOAA/NCEI Global Hourly','url':url,'status':r.status_code,'bytes':len(r.content),'rows':len(z)}

def fetch_2026_asos():
    # IEM ASOS/METAR archive for the same physical station KIKK; observational data only.
    url=(
      'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=IKK'
      '&data=tmpf&data=dwpf&data=relh'
      '&year1=2026&month1=1&day1=1&year2=2026&month2=8&day2=27'
      '&tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes'
      '&missing=empty&trace=empty&direct=no&report_type=1&report_type=3&report_type=4'
    )
    r=req(url,120)
    if r.status_code!=200: raise RuntimeError(f'IEM status={r.status_code}')
    text=r.content.decode('utf-8','replace')
    if 'valid' not in text[:1000] or 'tmpf' not in text[:1000]: raise RuntimeError(f'Unexpected IEM response: {text[:500]}')
    d=pd.read_csv(io.StringIO(text),dtype=str,low_memory=False)
    z=pd.DataFrame({'obs_ts':pd.to_datetime(d['valid'],errors='coerce',utc=True)})
    tmpf=pd.to_numeric(d.get('tmpf'),errors='coerce'); dwpf=pd.to_numeric(d.get('dwpf'),errors='coerce')
    z['temp_c']=(tmpf-32)*5/9; z['dew_c']=(dwpf-32)*5/9
    z['rh_reported']=pd.to_numeric(d.get('relh'),errors='coerce')
    z['source']='IEM_ASOS_KIKK'; z['source_year']=2026
    return z.dropna(subset=['obs_ts']), {'year':2026,'source':'Iowa Environmental Mesonet ASOS/METAR archive, station KIKK','url':url,'status':r.status_code,'bytes':len(r.content),'rows':len(z)}

def std_p(h): return 101325.0*(1-2.25577e-5*h)**5.2559

def choose(valid):
    v=valid.reset_index(drop=True).copy(); v['obs_id']=np.arange(len(v),dtype=np.int64)
    a=v.copy(); a['target_hour']=a.obs_ts.dt.floor('h')
    b=v.copy(); b['target_hour']=b.obs_ts.dt.ceil('h')
    c=pd.concat([a,b],ignore_index=True).drop_duplicates(['obs_id','target_hour'])
    c['delta']=(c.obs_ts-c.target_hour).abs().dt.total_seconds()
    c=c[(c.delta<=1800)&c.target_hour.between(START_UTC,END_UTC)]
    c['exact']=c.delta.eq(0); c['complete']=c[['temp_c','dew_c','rh_pct']].notna().sum(axis=1)
    c=c.sort_values(['target_hour','exact','delta','complete','obs_ts'],ascending=[True,False,True,False,True],kind='mergesort')
    used=set(); rows=[]
    for th,g in c.groupby('target_hour',sort=True):
        for row in g.itertuples(index=False):
            if int(row.obs_id) not in used:
                used.add(int(row.obs_id)); rows.append(row); break
    return pd.DataFrame(rows).set_index('target_hour').sort_index() if rows else pd.DataFrame()

def main():
    assert len(TARGET)==131520
    psychrolib.SetUnitSystem(psychrolib.SI)
    cand=candidate_test(); parts=[]; dl=[]
    for y in range(2011,2026):
        z,m=fetch_noaa_year(y); parts.append(z); dl.append(m); print('NOAA',y,len(z),m['bytes'])
    z,m=fetch_2026_asos(); parts.append(z); dl.append(m); print('ASOS 2026',len(z),m['bytes'])
    obs=pd.concat(parts,ignore_index=True).sort_values('obs_ts').reset_index(drop=True)
    obs=obs[obs.obs_ts.between(START_UTC-pd.Timedelta(minutes=30),END_UTC+pd.Timedelta(minutes=30))].copy()
    rh=np.full(len(obs),np.nan)
    for i,(t,d) in enumerate(zip(obs.temp_c.to_numpy(float),obs.dew_c.to_numpy(float))):
        if np.isfinite(t) and np.isfinite(d) and d<=t+0.3:
            try: rh[i]=100*psychrolib.GetRelHumFromTDewPoint(float(t),float(d))
            except Exception: pass
    obs['rh_pct']=rh
    # Use reported ASOS RH only to cross-check; output RH remains consistently calculated from T/dew for all years.
    valid=obs[obs.temp_c.notna()&obs.dew_c.notna()&obs.rh_pct.between(0,100)].copy()
    sel=choose(valid)
    cols=['obs_ts','temp_c','dew_c','rh_pct','source','source_year']
    if sel.empty:
        sm=pd.DataFrame(columns=cols); sm.index=pd.DatetimeIndex([],tz='UTC',name='target_hour')
    else: sm=sel[cols]
    o=pd.DataFrame(index=TARGET); o.index.name='target_hour'; o=o.join(sm,how='left'); ok=o.obs_ts.notna()
    p=std_p(ELEVATION_M); wb=np.full(len(o),np.nan)
    for i,(t,r) in enumerate(zip(o.temp_c.to_numpy(float),o.rh_pct.to_numpy(float))):
        if np.isfinite(t) and np.isfinite(r): wb[i]=psychrolib.GetTWetBulbFromRelHum(float(t),float(r)/100,p)
    local=o.index.tz_convert(LOCAL_TZ)
    final=pd.DataFrame({
      'Date':local.strftime('%Y-%m-%d'),'Time':local.strftime('%H:%M:%S'),'Time_Zone':[x.tzname() for x in local],
      'Wet_Bulb_Temperature_F':wb*1.8+32,'Dry_Bulb_Air_Temperature_F':o.temp_c.to_numpy(float)*1.8+32,
      'Relative_Humidity_pct':o.rh_pct.to_numpy(float),'Dew_Point_Temperature_F':o.dew_c.to_numpy(float)*1.8+32,
      'Timestamp_UTC':o.index.strftime('%Y-%m-%dT%H:%M:%SZ'),'Observation_Timestamp_UTC':o.obs_ts.dt.strftime('%Y-%m-%dT%H:%M:%SZ').to_numpy(),
      'Wet_Bulb_Method':np.where(ok,'CALCULATED_FROM_OBSERVED_DRY_BULB_AND_DEW_POINT','MISSING'),
      'Data_Status':np.where(ok,'OK','MISSING_OBSERVATION'),'Source_Station_ID':GHCN_ID,'Source_Station_Name':STATION_NAME,
      'Source_Dataset':o['source'].to_numpy()
    })
    nums=['Wet_Bulb_Temperature_F','Dry_Bulb_Air_Temperature_F','Relative_Humidity_pct','Dew_Point_Temperature_F']
    for c in nums: final[c]=pd.to_numeric(final[c],errors='coerce').round(2)
    final.to_csv(OUT,index=False,lineterminator='\n'); check=pd.read_csv(OUT,dtype=str)
    wet=final.Wet_Bulb_Temperature_F; dry=final.Dry_Bulb_Air_Temperature_F; dew=final.Dew_Point_Temperature_F; r=final.Relative_Humidity_pct
    qa={
      'source_summary':'NOAA/NCEI Global Hourly 2011-2025 plus KIKK ASOS/METAR observational archive for 2026',
      'candidate_2024_url_tests':cand,'download_log':dl,
      'total_target_rows':len(final),'populated_rows':int(ok.sum()),'missing_rows':int((~ok).sum()),'coverage_pct':round(float(ok.mean()*100),4),
      'first_row_local':final[['Date','Time','Time_Zone']].iloc[0].to_dict(),'last_row_local':final[['Date','Time','Time_Zone']].iloc[-1].to_dict(),
      'first_target_utc':final.Timestamp_UTC.iloc[0],'last_target_utc':final.Timestamp_UTC.iloc[-1],
      'first_populated_target_utc':final.loc[ok,'Timestamp_UTC'].iloc[0] if ok.any() else None,'last_populated_target_utc':final.loc[ok,'Timestamp_UTC'].iloc[-1] if ok.any() else None,
      'last_available_observation_utc':obs.obs_ts.max().strftime('%Y-%m-%dT%H:%M:%SZ') if len(obs) else None,
      'duplicate_target_timestamps':int(final.Timestamp_UTC.duplicated().sum()),'source_observation_reuse':int(final.loc[ok,'Observation_Timestamp_UTC'].duplicated().sum()),
      'rh_outside_0_100':int(((r<0)|(r>100)).sum()),'dewpoint_gt_drybulb':int((dew>dry+.1).sum()),'wetbulb_gt_drybulb':int((wet>dry+.1).sum()),'wetbulb_lt_dewpoint':int((wet<dew-.1).sum()),
      'csv_rows':len(check),'csv_exact_columns_order':list(check.columns)==list(final.columns),'standard_pressure_pa':round(p,2),
      'source_counts':final.Source_Dataset.value_counts(dropna=False).to_dict()
    }
    QA.write_text(json.dumps(qa,indent=2,default=str),encoding='utf-8')
    print(json.dumps({k:v for k,v in qa.items() if k not in ['download_log','candidate_2024_url_tests']},indent=2,default=str))
    assert len(final)==131520 and len(check)==131520 and qa['duplicate_target_timestamps']==0 and qa['source_observation_reuse']==0
    assert qa['rh_outside_0_100']==0 and qa['dewpoint_gt_drybulb']==0 and qa['wetbulb_gt_drybulb']==0 and qa['wetbulb_lt_dewpoint']==0 and qa['csv_exact_columns_order']

if __name__=='__main__': main()
