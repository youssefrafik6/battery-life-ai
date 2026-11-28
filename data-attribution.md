# Telemetry data and attribution

The experiment uses the five English-language telemetry files in the
[Zenodo record](https://zenodo.org/records/13982080) by Daniel Flores-Martin,
Sergio Laso Mangas, and Juan Luis Herrera. The archive is published under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- `OnePlus_LE2113_en.csv`
- `OnePlus_LE2123_en.csv`
- `POCO_M2102J20SG_en.csv`
- `Samsung_SM-A226B_en.csv`
- `Samsung_SM-G991B_en.csv`

Run `batterylife download` to fetch the original archive and verify the
publisher's MD5 checksum. Run `batterylife prepare` to regenerate the
interval diagnostics. The bundled explorer retains only device model,
quality category, charge, and the plotted rate and usage fields. It
expresses time as hours since each device's first observation.

After preparing the interval file, run `python scripts/export_explorer.py`
to regenerate the reduced table.
