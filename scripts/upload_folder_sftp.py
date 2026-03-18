from __future__ import annotations

import argparse
import os
from pathlib import Path

import paramiko


def sftp_mkdir_p(sftp: paramiko.SFTPClient, remote_directory: str) -> None:
    parts = [p for p in remote_directory.replace("\\", "/").split("/") if p]
    if remote_directory.startswith("/"):
        current = "/"
    else:
        current = ""
    for part in parts:
        if current in {"", "/"}:
            current = f"{current}{part}" if current == "/" else part
        else:
            current = f"{current}/{part}"
        try:
            sftp.stat(current)
        except IOError:
            sftp.mkdir(current)


def collect_files(local_root: Path) -> list[Path]:
    files: list[Path] = []
    for root, _, filenames in os.walk(local_root):
        for name in filenames:
            files.append(Path(root) / name)
    return files


def remote_path_join(base: str, rel: str) -> str:
    base_norm = base.replace("\\", "/").rstrip("/")
    rel_norm = rel.replace("\\", "/").lstrip("/")
    if not base_norm:
        return rel_norm
    return f"{base_norm}/{rel_norm}"


def upload_folder(
    host: str,
    port: int,
    username: str,
    password: str,
    local_dir: Path,
    remote_dir: str,
) -> None:
    local_dir = local_dir.resolve()
    if not local_dir.exists() or not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    files = collect_files(local_dir)
    total_bytes = sum(f.stat().st_size for f in files)
    print(f"[upload] local_dir={local_dir}")
    print(f"[upload] file_count={len(files)} total_gb={total_bytes / (1024 ** 3):.3f}")
    print(f"[upload] remote_dir={remote_dir}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=username,
        password=password,
        timeout=30,
        banner_timeout=60,
        auth_timeout=60,
    )
    transport = client.get_transport()
    if transport is not None:
        transport.set_keepalive(30)
    sftp = client.open_sftp()

    try:
        sftp_mkdir_p(sftp, remote_dir)
        uploaded = 0
        skipped = 0
        transferred = 0
        report_every = max(1, len(files) // 20)  # ~5%
        for idx, local_file in enumerate(files, start=1):
            rel = local_file.relative_to(local_dir).as_posix()
            remote_file = remote_path_join(remote_dir, rel)
            remote_parent = remote_file.rsplit("/", 1)[0]
            sftp_mkdir_p(sftp, remote_parent)

            local_size = local_file.stat().st_size
            skip = False
            try:
                rstat = sftp.stat(remote_file)
                if int(rstat.st_size) == int(local_size):
                    skip = True
            except IOError:
                pass

            if skip:
                skipped += 1
                transferred += local_size
            else:
                sftp.put(str(local_file), remote_file)
                uploaded += 1
                transferred += local_size

            if idx % report_every == 0 or idx == len(files):
                pct = 100.0 * transferred / max(1, total_bytes)
                print(
                    f"[upload] {idx}/{len(files)} files | "
                    f"uploaded={uploaded} skipped={skipped} | "
                    f"progress={pct:.2f}%"
                )
    finally:
        sftp.close()
        client.close()

    print("[upload] done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a local folder to remote server via SFTP.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--remote-dir", required=True)
    args = parser.parse_args()

    upload_folder(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        local_dir=Path(args.local_dir),
        remote_dir=args.remote_dir,
    )


if __name__ == "__main__":
    main()
