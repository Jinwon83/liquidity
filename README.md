# Liquidity Hegemony Dashboard (Serverless)

KOSPI–NASDAQ / 달러 유동성 가설 검증 대시보드를 **유료 서버·DB 임대 없이**  
GitHub 무료 기능만으로 운영합니다.

## 아키텍처

```
FRED / yfinance /(선택) LS
        │
        ▼
 GitHub Actions  (무료 ETL 워커)
        │
        ├─► data/processed/*.csv
        ├─► docs/data/*.json   ← "서버리스 DB" (리포가 저장소)
        └─► (선택) Oracle Cloud Always Free 동기화
        │
        ▼
 GitHub Pages  (무료 정적 호스팅)
        │
        ▼
 브라우저 (Plotly.js 대시보드)
```

| 역할 | 사용 기능 | 비용 |
|------|-----------|------|
| 웹 호스팅 | GitHub Pages | 무료 (public 리포) |
| 스케줄/ETL | GitHub Actions | 무료 할당량 |
| 데이터 저장 | `docs/data/*.json` 커밋 | 무료 |
| (선택) 외부 DB | Oracle Always Free | 무료 티어 / 기존 계정 |

> Streamlit 로컬 대시보드(`dashboard.py`)도 유지합니다.  
> **공개 웹사이트는 `docs/` 정적 사이트**입니다.

## 로컬 실행

```bash
python -m venv .venv312
.venv312\Scripts\pip install -r requirements.txt
copy .env.example .env   # 키 입력

python run_stage12.py
python correlation_analysis.py
python scripts/export_pages_data.py

# 로컬 Streamlit
.venv312\Scripts\streamlit run dashboard.py

# 정적 사이트 미리보기 (docs)
# VS Code Live Server 또는: python -m http.server 8080 -d docs
```

## GitHub 설정

1. 이 리포를 **Public** 으로 유지 (무료 Pages 조건)
2. **Settings → Pages → Source = GitHub Actions**
3. Secrets (Settings → Secrets and variables → Actions)
   - `FRED_API_KEY` (권장)
   - (선택) `LS_APP_KEY`, `LS_APP_SECRET_KEY`
   - (선택 Oracle) `ORACLE_MODE=object_storage|atp` + 관련 시크릿
4. Actions 탭에서 **Update data & deploy Pages** → Run workflow

대시보드 URL 예:
`https://jinwon83.github.io/liquidity/`

## Oracle 업로드 (선택)

기본 데이터 경로는 GitHub JSON입니다. 기존 Oracle Cloud Always Free가 있을 때만:

```bash
# Object Storage
set ORACLE_MODE=object_storage
set OCI_NAMESPACE=...
set OCI_BUCKET=...
python scripts/sync_oracle.py

# Autonomous DB
set ORACLE_MODE=atp
set ORACLE_USER=...
set ORACLE_PASSWORD=...
set ORACLE_DSN=...
python scripts/sync_oracle.py
```

Pages 프론트는 동일 출처의 `docs/data`를 읽습니다. Oracle을 쓰더라도  
Actions가 JSON을 리포에 커밋해 Pages가 동작하도록 두는 구성을 권장합니다.

## 주요 파일

- `docs/` — GitHub Pages 사이트
- `scripts/export_pages_data.py` — CSV → JSON
- `scripts/sync_oracle.py` — 선택적 Oracle 동기화
- `.github/workflows/update-and-deploy.yml` — 수집·분석·배포
- `dashboard.py` — 로컬 Streamlit
