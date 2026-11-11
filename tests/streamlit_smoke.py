import time, sys
from streamlit.testing.v1 import AppTest
pages = ["pages/1_overview.py","pages/2_forecasting.py","pages/3_safety_stock.py","pages/4_monte_carlo.py",
         "pages/5_optimisation.py","pages/6_inventory.py","pages/7_reports.py","pages/8_data.py"]
at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "apps" / "streamlit" / "app.py"), default_timeout=300)
t=time.time(); at.run(); print("first run", round(time.time()-t,1), "exceptions:", [e.value for e in at.exception])
for p in pages:
    t=time.time(); at.switch_page(p); at.run()
    print(p, round(time.time()-t,1), "exc:", [e.message[:300] for e in at.exception], "errors:", [e.value[:200] for e in at.error])
