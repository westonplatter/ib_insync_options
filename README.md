# IB Insync Options
Define, download, persist, and analyze options data for IBKR via `ib-insync`.

## Examples
See [example_downloads_for_es_gc_cl.ipynb](example_downloads_for_es_gc_cl.ipynb).

## Local Development
This repo uses uv for dependency management. See their docs for installing, https://github.com/astral-sh/uv.

```
uv install
```

The example notebook requires a local postgres database.
```
psql -U postgres
create database ib_insync_options_dev;
```

## License
BSD-3 Clause. See [LICENSE](LICENSE) file for details.
