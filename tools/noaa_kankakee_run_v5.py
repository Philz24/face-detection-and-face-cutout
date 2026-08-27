from pathlib import Path

source_path = Path(__file__).with_name('noaa_kankakee_build_v4.py')
src = source_path.read_text(encoding='utf-8')
# Pandas 3 requires the boolean mask to share the target DataFrame index.
# final has a RangeIndex while ok inherits the UTC DatetimeIndex, so use positional boolean arrays.
src = src.replace("final.loc[ok,'Timestamp_UTC']", "final.loc[ok.to_numpy(),'Timestamp_UTC']")
src = src.replace("final.loc[ok,'Observation_Timestamp_UTC']", "final.loc[ok.to_numpy(),'Observation_Timestamp_UTC']")
src = src.replace("if __name__=='__main__': main()", "")
namespace = {'__name__': 'noaa_kankakee_build_v4_patched'}
exec(compile(src, str(source_path), 'exec'), namespace)
namespace['main']()
