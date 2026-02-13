"""
Data Tools - file discovery, loading, profiling, cleaning, charts, ML.
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from datetime import datetime

import pandas as pd


def _artifact_line(payload: dict) -> str:
    """Encode a renderable artifact payload as a single-line marker.

    Frontend will detect lines starting with `ADK_ARTIFACT:` and render accordingly.
    """

    try:
        import json

        return "ADK_ARTIFACT:" + json.dumps(payload, ensure_ascii=False)
    except Exception:
        return ""


_MAX_TOOL_CHARS = 8000


def _truncate_tool_output(text: str, limit: int = _MAX_TOOL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[TRUNCATED: output too long — narrow your request or load a specific file]"

DATA_EXTENSIONS = [".csv", ".tsv", ".xlsx", ".xls"]
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "datasets",
)


def _posix(path: str) -> str:
    """Convert Windows paths to forward slashes for JSON safety."""
    return path.replace("\\", "/")


def _get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_path(file_path: str) -> str | None:
    """Find data file across multiple locations."""
    file_path = file_path.replace("/", os.sep).replace("\\", os.sep)

    for base in [file_path, os.path.join(DATA_DIR, file_path),
                 os.path.join(_get_project_root(), file_path),
                 os.path.join(os.getcwd(), file_path)]:
        if os.path.isfile(base):
            return _posix(os.path.abspath(base))

    basename = os.path.basename(file_path)
    for root, _dirs, files in os.walk(_get_project_root()):
        if basename in files:
            return _posix(os.path.abspath(os.path.join(root, basename)))
    return None


def _read_dataframe(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, sep="\t", engine="python", encoding="latin1", errors="replace")


def list_data_files() -> str:
    """List available CSV/Excel files. Call this FIRST."""
    # Keep this intentionally narrow to avoid generating massive outputs that blow up LLM context.
    # We only scan:
    # - datasets/ (recursive)
    # - project root (non-recursive)
    search_dirs = [DATA_DIR, _get_project_root()]
    all_files = []
    seen = set()

    for directory in search_dirs:
        if not os.path.exists(directory):
            continue
        for ext in DATA_EXTENSIONS:
            pattern = f"**/*{ext}" if os.path.abspath(directory) == os.path.abspath(DATA_DIR) else f"*{ext}"
            for fp in glob.glob(os.path.join(directory, pattern), recursive=(pattern.startswith("**/"))):
                abs_path = os.path.abspath(fp)
                if abs_path in seen or any(part.startswith(".") for part in Path(abs_path).parts):
                    continue
                seen.add(abs_path)
                stat = os.stat(abs_path)
                all_files.append({
                    "path": _posix(abs_path),
                    "filename": os.path.basename(abs_path),
                    "size_kb": round(stat.st_size / 1024, 1),
                })

    all_files.sort(key=lambda f: f["filename"].lower())
    if not all_files:
        return "No data files found. Place CSV/Excel files in project root or datasets/ folder."

    max_files = 50
    omitted = max(0, len(all_files) - max_files)
    shown = all_files[:max_files]

    lines = [f"Found {len(all_files)} file(s) (showing {len(shown)}):"]
    for i, f in enumerate(shown, 1):
        lines.append(f"  {i}. {f['filename']} ({f['size_kb']} KB) -> {f['path']}")
    if omitted:
        lines.append(f"  ... ({omitted} more not shown)")
    lines.append("\nUse load_dataset(file_path) with the full path.")
    return _truncate_tool_output("\n".join(lines))


def load_dataset(file_path: str) -> str:
    """Load file and return preview."""
    resolved = _resolve_path(file_path)
    if not resolved:
        return _truncate_tool_output(f"Error: '{file_path}' not found.\n{list_data_files()}")

    try:
        df = _read_dataframe(resolved)
    except Exception as e:
        return f"Error reading '{resolved}': {e}"

    max_cols = 50
    cols = df.columns.tolist()
    shown_cols = cols[:max_cols]
    more_cols = max(0, len(cols) - max_cols)

    lines = [
        f"Loaded: {os.path.basename(resolved)}",
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        "\nColumns (up to 50): " + ", ".join(f"{c}({df[c].dtype})" for c in shown_cols)
        + (f" ... (+{more_cols} more)" if more_cols else ""),
        f"\nPreview (first 5 rows):\n{df[shown_cols].head(5).to_string(index=False, max_colwidth=30)}",
    ]

    preview_df = df[shown_cols].head(20).copy()
    preview_df = preview_df.where(pd.notna(preview_df), "")
    artifact = _artifact_line(
        {
            "kind": "table",
            "title": f"Preview: {os.path.basename(resolved)}",
            "columns": [str(c) for c in preview_df.columns.tolist()],
            "rows": preview_df.astype(str).to_dict(orient="records"),
        }
    )

    out = "\n".join(lines)
    if artifact:
        out += "\n\n" + artifact
    return _truncate_tool_output(out)


def profile_dataset(file_path: str) -> str:
    """Profile dataset: types, nulls, stats, correlations."""
    resolved = _resolve_path(file_path)
    if not resolved:
        return _truncate_tool_output(f"Error: '{file_path}' not found.")

    try:
        df = _read_dataframe(resolved)
    except Exception as e:
        return f"Error: {e}"

    max_cols = 80
    cols = df.columns.tolist()
    shown_cols = cols[:max_cols]
    more_cols = max(0, len(cols) - max_cols)

    lines = [
        f"Profile: {os.path.basename(resolved)}",
        f"Shape: {df.shape[0]} rows x {df.shape[1]} cols | Duplicates: {df.duplicated().sum()}",
    ]

    lines.append("\nColumns:")
    for col in shown_cols:
        null_pct = df[col].isnull().mean() * 100
        unique = df[col].nunique()
        lines.append(f"  {col}: {df[col].dtype}, {unique} unique, {null_pct:.1f}% null")
    if more_cols:
        lines.append(f"  ... (+{more_cols} more columns not shown)")

    numeric = df[shown_cols].select_dtypes(include=["number"]).columns
    if len(numeric) > 0:
        numeric_show = list(numeric[:20])
        lines.append(f"\nNumeric Summary (up to 20 cols):\n{df[numeric_show].describe().round(2).to_string()}")
        if len(numeric) > 1:
            corr = df[numeric_show].corr()
            pairs = [(c1, c2, corr.loc[c1, c2]) for i, c1 in enumerate(numeric_show)
                     for c2 in numeric_show[i+1:] if pd.notna(corr.loc[c1, c2])]
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            if pairs[:3]:
                lines.append("\nTop correlations:")
                for c1, c2, v in pairs[:3]:
                    lines.append(f"  {c1}-{c2}: {v:.3f}")

    cat = df[shown_cols].select_dtypes(include=["object", "category"]).columns
    if len(cat) > 0:
        lines.append("\nCategorical (top values):")
        for col in cat[:5]:
            top = df[col].value_counts().head(3)
            lines.append(f"  {col}: {dict(top)}")

    return _truncate_tool_output("\n".join(lines))


def clean_dataset(file_path: str, drop_dupes: bool = True, fill_missing: bool = True) -> str:
    """Clean dataset: remove dupes, fill nulls. Saves to _cleaned.csv."""
    resolved = _resolve_path(file_path)
    if not resolved:
        return _truncate_tool_output(f"Error: '{file_path}' not found.")

    try:
        df = _read_dataframe(resolved)
    except Exception as e:
        return f"Error: {e}"

    before = df.shape
    actions = []

    if drop_dupes:
        n = df.duplicated().sum()
        if n > 0:
            df = df.drop_duplicates(ignore_index=True)
            actions.append(f"Removed {n} duplicates")

    if fill_missing:
        filled = 0
        for col in df.columns:
            nulls = df[col].isnull().sum()
            if nulls > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode_vals = df[col].mode(dropna=True)
                    df[col] = df[col].fillna(mode_vals.iloc[0] if len(mode_vals) > 0 else "")
                filled += nulls
        if filled > 0:
            actions.append(f"Filled {filled} nulls")

    clean_name = Path(resolved).stem + "_cleaned.csv"
    os.makedirs(DATA_DIR, exist_ok=True)
    clean_path = os.path.join(DATA_DIR, clean_name)
    df.to_csv(clean_path, index=False)

    return _truncate_tool_output(
        (
            f"Cleaned: {before[0]}x{before[1]} -> {df.shape[0]}x{df.shape[1]}\n"
            f"Actions: {'; '.join(actions) if actions else 'None needed'}\n"
            f"Saved: {_posix(clean_path)}"
        )
    )


def analyze_trends(file_path: str) -> str:
    """Analyze dataset for trends, outliers, patterns."""
    resolved = _resolve_path(file_path)
    if not resolved:
        return _truncate_tool_output(f"Error: '{file_path}' not found.")

    try:
        df = _read_dataframe(resolved)
    except Exception as e:
        return f"Error: {e}"

    lines = [f"Trend Analysis: {os.path.basename(resolved)} ({df.shape[0]} rows)"]
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric:
        lines.append("\nOutliers (IQR method):")
        for col in numeric[:5]:
            s = df[col].dropna()
            if len(s) > 0:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                outliers = ((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum()
                lines.append(f"  {col}: {outliers} outliers ({outliers/len(s)*100:.1f}%)")

    if cat and numeric:
        lines.append("\nGroup Means:")
        for c in cat[:3]:
            if df[c].nunique() <= 15:
                try:
                    g = df.groupby(c)[numeric[:3]].mean().round(2)
                    lines.append(f"  By {c}:\n{g.to_string().replace(chr(10), chr(10)+'    ')}")
                except Exception:
                    pass

    if len(numeric) > 1:
        corr = df[numeric].corr()
        strong = [(c1, c2, corr.loc[c1, c2]) for i, c1 in enumerate(numeric)
                  for c2 in numeric[i+1:] if abs(corr.loc[c1, c2]) > 0.7]
        if strong:
            lines.append("\nStrong correlations (|r| > 0.7):")
            for c1, c2, v in strong[:5]:
                lines.append(f"  {c1} - {c2}: {v:.3f}")

    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        worst = df.isnull().sum().idxmax()
        lines.append(f"\nMissing data: {total_missing} total, worst column: {worst}")

    out = "\n".join(lines) if len(lines) > 1 else lines[0] + "\nNo strong patterns detected."
    return _truncate_tool_output(out)


def add_datetime_features(file_path: str, min_parse_ratio: float = 0.8) -> str:
    """Add year/month/dayofweek features from datetime columns. Saves to _features.csv."""
    resolved = _resolve_path(file_path)
    if not resolved:
        return _truncate_tool_output(f"Error: '{file_path}' not found.")

    try:
        df = _read_dataframe(resolved)
    except Exception as e:
        return f"Error: {e}"

    created = {}
    out = df.copy()

    for col in list(out.columns):
        try:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                dt = out[col]
            else:
                parsed = pd.to_datetime(out[col], errors="coerce")
                if parsed.notna().mean() >= min_parse_ratio:
                    dt = parsed
                else:
                    continue

            feats = []
            for suffix, accessor in [("_year", "year"), ("_month", "month"), ("_dow", "dayofweek")]:
                name = f"{col}{suffix}"
                if name not in out.columns:
                    out[name] = getattr(dt.dt, accessor)
                    feats.append(suffix.replace("_", ""))
            if feats:
                created[col] = feats
        except Exception:
            continue

    if not created:
        return _truncate_tool_output("No datetime columns detected.")

    enhanced_name = Path(resolved).stem + "_features.csv"
    os.makedirs(DATA_DIR, exist_ok=True)
    enhanced_path = os.path.join(DATA_DIR, enhanced_name)
    out.to_csv(enhanced_path, index=False)

    lines = ["Datetime features added:"]
    for col, feats in created.items():
        lines.append(f"  {col}: {', '.join(feats)}")
    lines.append(f"\nSaved: {_posix(enhanced_path)}")
    return _truncate_tool_output("\n".join(lines))


def train_small_model(file_path: str, target: str, problem_type: str = "auto") -> str:
    """Train quick baseline model (Logistic/Linear Regression)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression, LinearRegression

    resolved = _resolve_path(file_path)
    if not resolved:
        return _truncate_tool_output(f"Error: '{file_path}' not found.")

    try:
        df = _read_dataframe(resolved)
    except Exception as e:
        return f"Error: {e}"

    if target not in df.columns:
        return _truncate_tool_output(f"Error: target '{target}' not found. Columns: {', '.join(df.columns[:80])}")

    X, y = df.drop(columns=[target]), df[target]
    if problem_type == "auto":
        problem_type = "classification" if y.nunique() <= 20 else "regression"

    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

    model = LogisticRegression(max_iter=500) if problem_type == "classification" else LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = Pipeline([("prep", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    if problem_type == "classification":
        acc = accuracy_score(y_test, preds)
        return _truncate_tool_output(f"Classification (Logistic): Accuracy = {acc:.3f} ({acc*100:.1f}%)")
    else:
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        return _truncate_tool_output(f"Regression (Linear): RMSE = {rmse:.3f}, R² = {r2:.3f}")


def build_chart(file_path: str, chart_type: str = "Bar", x: str | None = None,
                y: str | None = None, color: str | None = None, agg: str = "sum", top_k: int = 10) -> str:
    """Create Plotly chart and return a renderable artifact (no filesystem writes)."""
    import plotly.express as px
    import plotly.io as pio
    import json

    resolved = _resolve_path(file_path)
    if not resolved:
        return _truncate_tool_output(f"Error: '{file_path}' not found.")

    try:
        df = _read_dataframe(resolved)
    except Exception as e:
        return f"Error: {e}"

    if x is None or x not in df.columns:
        return _truncate_tool_output(f"Error: x column '{x}' not found. Columns: {', '.join(df.columns[:80])}")
    if y and y not in df.columns:
        return _truncate_tool_output(f"Error: y column '{y}' not found.")

    df2 = df.copy()
    fig = None

    if chart_type in ["Bar", "Line"] and y:
        if pd.api.types.is_string_dtype(df2[x]) or isinstance(df2[x].dtype, pd.CategoricalDtype):
            agg_df = df2.groupby(x, dropna=False)[y].agg(agg).reset_index().sort_values(y, ascending=False).head(int(top_k))
            fig = px.bar(agg_df, x=x, y=y, title=f"{y} by {x}") if chart_type == "Bar" else px.line(agg_df, x=x, y=y, title=f"{y} by {x}")
        else:
            fig = px.bar(df2, x=x, y=y, title=f"{y} vs {x}") if chart_type == "Bar" else px.line(df2, x=x, y=y, title=f"{y} vs {x}")
    elif chart_type == "Scatter" and y:
        # keep payload small for UI streaming
        max_points = 800
        if len(df2) > max_points:
            df2 = df2.sample(n=max_points, random_state=42)
        fig = px.scatter(df2, x=x, y=y, title=f"{y} vs {x}")
    elif chart_type == "Histogram":
        # pre-aggregate bins to avoid huge raw arrays in plotly JSON
        s = pd.to_numeric(df2[x], errors="coerce").dropna()
        if len(s) == 0:
            return _truncate_tool_output(f"Error: column '{x}' has no numeric values for histogram.")
        bins = 50
        counts, edges = pd.cut(s, bins=bins, retbins=True, include_lowest=True)
        bin_counts = counts.value_counts(sort=False)
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        hist_df = pd.DataFrame({"bin_center": centers, "count": bin_counts.values})
        fig = px.bar(hist_df, x="bin_center", y="count", title=f"Distribution of {x}")
    elif chart_type == "Pie":
        if y and pd.api.types.is_numeric_dtype(df2[y]):
            agg_df = df2.groupby(x, dropna=False)[y].sum().reset_index().sort_values(y, ascending=False).head(int(top_k))
            fig = px.pie(agg_df, names=x, values=y, title=f"{y} by {x}")
        else:
            vc = df2[x].value_counts().reset_index()
            vc.columns = [x, "count"]
            fig = px.pie(vc.head(int(top_k)), names=x, values="count", title=f"Distribution of {x}")

    if fig is None:
        return _truncate_tool_output(f"Could not build {chart_type} chart.")

    try:
        # Use Plotly's JSON serializer so payload is always JSON-safe.
        # fig.to_dict() can contain objects that stdlib json can't encode.
        spec = json.loads(pio.to_json(fig, remove_uids=True, pretty=False))
        if isinstance(spec, dict):
            layout = spec.get("layout")
            if isinstance(layout, dict) and "template" in layout:
                layout.pop("template", None)
    except Exception as e:
        return _truncate_tool_output(f"Error: failed to serialize chart: {e}")

    artifact = _artifact_line(
        {
            "kind": "chart_plotly",
            "title": f"{chart_type} chart",
            "spec": spec,
        }
    )

    out = f"{chart_type} chart generated."
    if artifact:
        # Do not truncate artifact payload; UI needs the full JSON to render charts.
        return out + "\n\n" + artifact
    return _truncate_tool_output(out)
