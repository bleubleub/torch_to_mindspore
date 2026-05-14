import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


OUTPUT_COLUMNS = ["可替换MindSpore接口", "其他替换方案", "备注"]


def _clean_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"不支持的输入文件格式: {path.suffix}，请使用 .xlsx、.xls 或 .csv")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"不支持的输出文件格式: {path.suffix}，请使用 .xlsx、.xls 或 .csv")


def _resolve_api_column(df: pd.DataFrame, requested_column: str) -> str:
    if requested_column in df.columns:
        return requested_column
    if len(df.columns) == 1:
        only_column = str(df.columns[0])
        print(f"未找到列 '{requested_column}'，输入表只有一列，将使用 '{only_column}' 作为接口列。")
        return only_column
    raise ValueError(f"输入表必须包含列 '{requested_column}'，当前列为: {list(df.columns)}")


def _build_mapping(mapping_file: Path) -> dict:
    mapping_df = pd.read_excel(mapping_file)
    required_columns = ["Torch_API", "MindSpore_API"]
    missing = [col for col in required_columns if col not in mapping_df.columns]
    if missing:
        raise ValueError(f"映射表缺少必要列: {missing}")

    mapping = {}
    for _, row in mapping_df.dropna(subset=["Torch_API"]).iterrows():
        torch_api = _clean_cell(row["Torch_API"])
        if not torch_api or torch_api in mapping:
            continue
        mapping[torch_api] = {
            "可替换MindSpore接口": _clean_cell(row.get("MindSpore_API", "")),
            "其他替换方案": _clean_cell(row.get("其他替代方案", "")),
            "备注": _clean_cell(row.get("备注", "")),
        }
    return mapping


def enrich_api_table(
    input_file: Path,
    output_file: Path,
    mapping_file: Path,
    api_column: str = "接口",
) -> None:
    input_df = _read_table(input_file)
    resolved_api_column = _resolve_api_column(input_df, api_column)
    mapping = _build_mapping(mapping_file)

    output_df = input_df.copy()
    for column in OUTPUT_COLUMNS:
        output_df[column] = ""

    matched_count = 0
    for index, api_value in output_df[resolved_api_column].items():
        torch_api = _clean_cell(api_value)
        info: Optional[dict] = mapping.get(torch_api)
        if info is None:
            if torch_api:
                output_df.at[index, "备注"] = "映射表未找到该接口"
            continue

        matched_count += 1
        for column in OUTPUT_COLUMNS:
            output_df.at[index, column] = info[column]

    _write_table(output_df, output_file)
    print(f"处理完成: {output_file}")
    print(f"输入接口数: {len(output_df)}，匹配映射数: {matched_count}，未匹配数: {len(output_df) - matched_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 torch_to_mindspore_mapping.xlsx 为 Torch API 表追加映射信息。")
    parser.add_argument("input_file", help="输入表格路径，默认读取名为 '接口' 的列。")
    parser.add_argument(
        "-o",
        "--output",
        help="输出表格路径。不指定时直接写回输入表格。",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        default="torch_to_mindspore_mapping.xlsx",
        help="Torch 到 MindSpore 的映射表路径。",
    )
    parser.add_argument(
        "--api-column",
        default="接口",
        help="输入表中的 Torch API 列名，默认是 '接口'。",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file

    enrich_api_table(
        input_file=input_file,
        output_file=output_file,
        mapping_file=Path(args.mapping),
        api_column=args.api_column,
    )


if __name__ == "__main__":
    main()
