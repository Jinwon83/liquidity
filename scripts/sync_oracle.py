"""
선택적 Oracle 업로드.

무료 서버리스 기본 경로는 GitHub 리포의 docs/data/*.json 입니다.
Oracle Cloud Always Free (Object Storage 또는 Autonomous DB)를 쓰는 경우에만
환경변수를 설정하고 이 스크립트를 Actions/로컬에서 실행하세요.

환경변수:
  ORACLE_MODE=object_storage|atp|off   (기본 off)
  # Object Storage (OCI)
  OCI_CONFIG_FILE / OCI_PROFILE
  OCI_NAMESPACE, OCI_BUCKET, OCI_PREFIX=liquidity/
  # 또는 ATP (oracledb thick/thin)
  ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "data"


def upload_object_storage() -> None:
    try:
        import oci  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "oci 패키지 필요: pip install oci\n"
            "또는 ORACLE_MODE=off 로 GitHub JSON만 사용하세요."
        ) from exc

    namespace = os.environ["OCI_NAMESPACE"]
    bucket = os.environ["OCI_BUCKET"]
    prefix = os.environ.get("OCI_PREFIX", "liquidity/")
    config = oci.config.from_file(
        os.environ.get("OCI_CONFIG_FILE", str(Path.home() / ".oci" / "config")),
        os.environ.get("OCI_PROFILE", "DEFAULT"),
    )
    client = oci.object_storage.ObjectStorageClient(config)
    for path in sorted(DATA_DIR.glob("*.json")):
        object_name = f"{prefix}{path.name}"
        with path.open("rb") as fh:
            client.put_object(namespace, bucket, object_name, fh)
        print(f"[OCI] uploaded {object_name}")


def upload_atp() -> None:
    try:
        import oracledb  # type: ignore
    except ImportError as exc:
        raise SystemExit("oracledb 패키지 필요: pip install oracledb") from exc

    user = os.environ["ORACLE_USER"]
    password = os.environ["ORACLE_PASSWORD"]
    dsn = os.environ["ORACLE_DSN"]
    conn = oracledb.connect(user=user, password=password, dsn=dsn)
    cur = conn.cursor()
    cur.execute(
        """
        BEGIN
          EXECUTE IMMEDIATE '
            CREATE TABLE liquidity_json (
              name VARCHAR2(128) PRIMARY KEY,
              payload CLOB,
              updated_at TIMESTAMP DEFAULT SYSTIMESTAMP
            )';
        EXCEPTION
          WHEN OTHERS THEN
            IF SQLCODE != -955 THEN RAISE; END IF;
        END;
        """
    )
    for path in sorted(DATA_DIR.glob("*.json")):
        payload = path.read_text(encoding="utf-8")
        cur.execute(
            """
            MERGE INTO liquidity_json t
            USING (SELECT :name AS name FROM dual) s
            ON (t.name = s.name)
            WHEN MATCHED THEN UPDATE SET payload = :payload, updated_at = SYSTIMESTAMP
            WHEN NOT MATCHED THEN INSERT (name, payload) VALUES (:name, :payload)
            """,
            {"name": path.name, "payload": payload},
        )
        print(f"[ATP] upserted {path.name}")
    conn.commit()
    cur.close()
    conn.close()


def main() -> None:
    mode = os.environ.get("ORACLE_MODE", "off").strip().lower()
    if mode in ("", "off", "none", "github"):
        print("[SKIP] ORACLE_MODE=off — GitHub docs/data 가 데이터 소스입니다.")
        return
    if not DATA_DIR.exists() or not any(DATA_DIR.glob("*.json")):
        print("[ERR] docs/data/*.json 없음 — 먼저 export_pages_data.py 실행", file=sys.stderr)
        raise SystemExit(1)
    if mode in ("object_storage", "oci", "os"):
        upload_object_storage()
    elif mode in ("atp", "db", "autonomous"):
        upload_atp()
    else:
        raise SystemExit(f"알 수 없는 ORACLE_MODE={mode}")


if __name__ == "__main__":
    main()
