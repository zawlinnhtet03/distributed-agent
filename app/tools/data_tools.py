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
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _posix(path: str) -> str:
    """Convert Windows paths to forward slashes for JSON safety."""
    return path.replace("\\", "/")


def _get_project_root() -> str:
    return PROJECT_ROOT


def _resolve_path(file_path: str) -> str | None:
    """Find data file across multiple locations."""
    file_path = file_path.replace("/", os.sep).replace("\\", os.sep)

    for base in [
        file_path,
        os.path.join(DATA_DIR, file_path),
        os.path.join(_get_project_root(), file_path),
        os.path.join(os.getcwd(), file_path),
    ]:
        if os.path.isfile(base):
            return _posix(os.path.abspath(base))

    basename = os.path.basename(file_path)
    for root, _dirs, files in os.walk(_get_project_root()):
        if basename in files:
            return _posix(os.path.abspath(os.path.join(root, basename)))
    return None


def _read_dataframe(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in set(DATA_EXTENSIONS):
        raise ValueError(
            f"Unsupported data file extension: '{suffix or 'none'}'. "
            f"Supported: {', '.join(DATA_EXTENSIONS)}"
        )
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    try:
        return pd.read_csv(
            path,
            sep="\t",
            engine="python",
            encoding="latin1",
            encoding_errors="replace",
        )
    except TypeError:
        return pd.read_csv(path, sep="\t", engine="python", encoding="latin1")


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

    palette = px.colors.qualitative.Vivid

    if chart_type in ["Bar", "Line"] and y:
        if pd.api.types.is_string_dtype(df2[x]) or isinstance(df2[x].dtype, pd.CategoricalDtype):
            agg_df = df2.groupby(x, dropna=False)[y].agg(agg).reset_index().sort_values(y, ascending=False).head(int(top_k))
            if chart_type == "Bar":
                fig = px.bar(
                    agg_df,
                    x=x,
                    y=y,
                    color=(color if (color and color in agg_df.columns) else x),
                    title=f"{y} by {x}",
                    color_discrete_sequence=palette,
                )
            else:
                fig = px.line(
                    agg_df,
                    x=x,
                    y=y,
                    color=(color if (color and color in agg_df.columns) else None),
                    title=f"{y} by {x}",
                    color_discrete_sequence=palette,
                )
        else:
            if chart_type == "Bar":
                fig = px.bar(
                    df2,
                    x=x,
                    y=y,
                    color=(color if (color and color in df2.columns) else None),
                    title=f"{y} vs {x}",
                    color_discrete_sequence=palette,
                )
            else:
                fig = px.line(
                    df2,
                    x=x,
                    y=y,
                    color=(color if (color and color in df2.columns) else None),
                    title=f"{y} vs {x}",
                    color_discrete_sequence=palette,
                )
    elif chart_type == "Scatter" and y:
        # keep payload small for UI streaming
        max_points = 800
        if len(df2) > max_points:
            df2 = df2.sample(n=max_points, random_state=42)
        fig = px.scatter(
            df2,
            x=x,
            y=y,
            color=(color if (color and color in df2.columns) else None),
            title=f"{y} vs {x}",
            color_discrete_sequence=palette,
        )
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
        fig = px.bar(
            hist_df,
            x="bin_center",
            y="count",
            title=f"Distribution of {x}",
            color_discrete_sequence=palette,
        )
    elif chart_type == "Pie":
        if y and pd.api.types.is_numeric_dtype(df2[y]):
            agg_df = df2.groupby(x, dropna=False)[y].sum().reset_index().sort_values(y, ascending=False).head(int(top_k))
            fig = px.pie(
                agg_df,
                names=x,
                values=y,
                title=f"{y} by {x}",
                color_discrete_sequence=palette,
            )
        else:
            vc = df2[x].value_counts().reset_index()
            vc.columns = [x, "count"]
            fig = px.pie(
                vc.head(int(top_k)),
                names=x,
                values="count",
                title=f"Distribution of {x}",
                color_discrete_sequence=palette,
            )

    if fig is None:
        return _truncate_tool_output(f"Could not build {chart_type} chart.")

    fig.update_layout(
        title={"x": 0.5, "xanchor": "center"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
        legend={"title": "", "orientation": "h", "y": -0.25},
        margin={"l": 50, "r": 30, "t": 60, "b": 70},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.15)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.15)")
    fig.update_traces(marker={"line": {"width": 0}}, opacity=0.95)

    if chart_type == "Bar" and color is None:
        fig.update_layout(showlegend=False)

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


def auto_visualize(file_path: str, max_charts: int = 3) -> str:
    import json
    import plotly.express as px
    import plotly.io as pio

    resolved = _resolve_path(file_path)
    if not resolved:
        return _truncate_tool_output(f"Error: '{file_path}' not found.")

    try:
        df = _read_dataframe(resolved)
    except Exception as e:
        return f"Error: {e}"

    if df.shape[0] == 0 or df.shape[1] == 0:
        return _truncate_tool_output("Error: dataset is empty.")

    palette = px.colors.qualitative.Vivid

    def _style(fig):
        fig.update_layout(
            title={"x": 0.5, "xanchor": "center"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e5e7eb"},
            legend={"title": "", "orientation": "h", "y": -0.25},
            margin={"l": 50, "r": 30, "t": 60, "b": 70},
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.15)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.15)")
        fig.update_traces(marker={"line": {"width": 0}}, opacity=0.95)
        return fig

    def _spec(fig):
        spec = json.loads(pio.to_json(fig, remove_uids=True, pretty=False))
        if isinstance(spec, dict):
            layout = spec.get("layout")
            if isinstance(layout, dict) and "template" in layout:
                layout.pop("template", None)
        return spec

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [c for c in df.columns.tolist() if c not in numeric_cols]

    def _cardinality(col: str) -> int:
        try:
            return int(df[col].nunique(dropna=True))
        except Exception:
            return 10**9

    def _is_datetime_like(col: str) -> bool:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return True
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            return parsed.notna().mean() >= 0.8
        except Exception:
            return False

    datetime_cols = [c for c in df.columns.tolist() if c in non_numeric_cols and _is_datetime_like(c)]

    y = None
    if numeric_cols:
        try:
            y = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False).index[0]
        except Exception:
            y = numeric_cols[0]

    small_cats = [c for c in non_numeric_cols if 2 <= _cardinality(c) <= 15]
    best_cat = None
    if small_cats:
        best_cat = sorted(small_cats, key=lambda c: _cardinality(c))[0]

    figs: list[tuple[str, object]] = []

    if y is not None and best_cat is not None and len(figs) < int(max_charts):
        try:
            agg_df = df.groupby(best_cat, dropna=False)[y].mean().reset_index().sort_values(y, ascending=False)
            if len(agg_df) > 12:
                agg_df = agg_df.head(12)
            fig = px.bar(
                agg_df,
                x=best_cat,
                y=y,
                color=best_cat,
                title=f"Average {y} by {best_cat}",
                color_discrete_sequence=palette,
            )
            figs.append(("Key comparison", _style(fig)))
        except Exception:
            pass

    if y is not None and len(figs) < int(max_charts):
        try:
            s = pd.to_numeric(df[y], errors="coerce").dropna()
            if len(s) > 0:
                bins = 40
                counts, edges = pd.cut(s, bins=bins, retbins=True, include_lowest=True)
                bin_counts = counts.value_counts(sort=False)
                centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
                hist_df = pd.DataFrame({"bin_center": centers, "count": bin_counts.values})
                fig = px.bar(
                    hist_df,
                    x="bin_center",
                    y="count",
                    title=f"Distribution of {y}",
                    color_discrete_sequence=palette,
                )
                figs.append(("Distribution", _style(fig)))
        except Exception:
            pass

    if y is not None and datetime_cols and len(figs) < int(max_charts):
        dt_col = datetime_cols[0]
        try:
            dt = df[dt_col]
            if not pd.api.types.is_datetime64_any_dtype(dt):
                dt = pd.to_datetime(dt, errors="coerce")
            tmp = pd.DataFrame({dt_col: dt, y: pd.to_numeric(df[y], errors="coerce")}).dropna()
            if len(tmp) > 0:
                tmp["__date"] = tmp[dt_col].dt.date
                g = tmp.groupby("__date")[y].mean().reset_index().sort_values("__date")
                fig = px.line(
                    g,
                    x="__date",
                    y=y,
                    title=f"Average {y} over time",
                    color_discrete_sequence=palette,
                )
                figs.append(("Trend", _style(fig)))
        except Exception:
            pass

    if y is not None and len(numeric_cols) >= 2 and len(figs) < int(max_charts):
        x_num = next((c for c in numeric_cols if c != y), None)
        if x_num is not None:
            try:
                df2 = df.copy()
                df2[y] = pd.to_numeric(df2[y], errors="coerce")
                df2[x_num] = pd.to_numeric(df2[x_num], errors="coerce")
                df2 = df2.dropna(subset=[y, x_num])
                if len(df2) > 800:
                    df2 = df2.sample(n=800, random_state=42)
                color_col = best_cat if best_cat in df2.columns else None
                fig = px.scatter(
                    df2,
                    x=x_num,
                    y=y,
                    color=color_col,
                    title=f"{y} vs {x_num}",
                    color_discrete_sequence=palette,
                )
                figs.append(("Relationship", _style(fig)))
            except Exception:
                pass

    if not figs:
        return _truncate_tool_output("No suitable charts could be generated for this dataset.")

    lines = [f"Auto-visualization generated {len(figs)} chart(s) for: {os.path.basename(resolved)}"]
    out_parts: list[str] = []
    out_parts.append("\n".join(lines))

    for title, fig in figs[: int(max_charts)]:
        try:
            spec = _spec(fig)
        except Exception as e:
            out_parts.append(_truncate_tool_output(f"Error: failed to serialize chart '{title}': {e}"))
            continue
        artifact = _artifact_line({"kind": "chart_plotly", "title": title, "spec": spec})
        if artifact:
            out_parts.append(artifact)

    return "\n\n".join(out_parts)
