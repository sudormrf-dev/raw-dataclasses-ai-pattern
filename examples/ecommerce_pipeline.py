"""E-commerce LLM pipeline: extract, classify, SEO, and store — all with raw dataclasses.

Demonstrates a four-stage pipeline with simulated LLM responses and SQLite in-memory
storage using only stdlib (dataclasses, sqlite3, json) — no ORM, no framework.

Run::

    python examples/ecommerce_pipeline.py
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field

from patterns.pipeline_stages import (
    PipelineInput,
    PipelineOutput,
    PostprocessConfig,
    PreprocessConfig,
    RawLLMResponse,
    postprocess,
    preprocess,
)
from patterns.result_types import Err, Ok


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RawProduct:
    """Unstructured product description from a merchant feed.

    Attributes:
        raw_text: Free-form product description.
        source_id: Merchant-assigned identifier.
    """

    raw_text: str
    source_id: str = ""


@dataclass
class ExtractedProduct:
    """Structured product extracted from a raw description.

    Attributes:
        name: Product name.
        brand: Brand name (empty string if unknown).
        raw_price: Price string as found in text (e.g. "$29.99").
        features: Key product features as a list.
        source_id: Propagated from :class:`RawProduct`.
    """

    name: str
    brand: str
    raw_price: str
    features: list[str]
    source_id: str = ""


@dataclass
class ClassifiedProduct:
    """Product with category, normalised price, and availability flag.

    Attributes:
        name: Product name.
        brand: Brand name.
        category: Top-level category (e.g. "Electronics").
        price_usd: Parsed price in USD (0.0 if not parseable).
        in_stock: True when the description implies availability.
        features: Carried from :class:`ExtractedProduct`.
        source_id: Propagated identifier.
    """

    name: str
    brand: str
    category: str
    price_usd: float
    in_stock: bool
    features: list[str]
    source_id: str = ""


@dataclass
class SEOProduct:
    """Product enriched with an SEO-optimised description.

    Attributes:
        name: Product name.
        brand: Brand.
        category: Category.
        price_usd: Price in USD.
        in_stock: Availability flag.
        seo_title: Title-tag-ready string (<60 chars).
        seo_description: Meta description (<160 chars).
        source_id: Propagated identifier.
    """

    name: str
    brand: str
    category: str
    price_usd: float
    in_stock: bool
    seo_title: str
    seo_description: str
    source_id: str = ""


# ---------------------------------------------------------------------------
# Simulated LLM responses (static strings — no API calls)
# ---------------------------------------------------------------------------

_EXTRACTION_RESPONSES: dict[str, str] = {
    "p001": json.dumps(
        {
            "name": "UltraSound Pro 3000",
            "brand": "SonicTech",
            "raw_price": "$149.99",
            "features": ["Noise-cancelling", "40h battery", "Foldable", "USB-C charging"],
        }
    ),
    "p002": json.dumps(
        {
            "name": "EcoBottle 1L",
            "brand": "GreenSip",
            "raw_price": "$24.95",
            "features": ["BPA-free", "Keeps cold 24h", "Leak-proof lid", "Dishwasher safe"],
        }
    ),
    "p003": json.dumps(
        {
            "name": "LuminaDesk LED Lamp",
            "brand": "BrightSpace",
            "raw_price": "$39.00",
            "features": ["5 brightness levels", "USB port", "Touch control", "Eye-care mode"],
        }
    ),
}

_CLASSIFICATION_RESPONSES: dict[str, str] = {
    "p001": json.dumps({"category": "Electronics", "price_usd": 149.99, "in_stock": True}),
    "p002": json.dumps({"category": "Sports & Outdoors", "price_usd": 24.95, "in_stock": True}),
    "p003": json.dumps({"category": "Home & Office", "price_usd": 39.00, "in_stock": False}),
}

_SEO_RESPONSES: dict[str, str] = {
    "p001": json.dumps(
        {
            "seo_title": "SonicTech UltraSound Pro 3000 Headphones | 40h Battery",
            "seo_description": (
                "Experience studio-quality audio with the SonicTech UltraSound Pro 3000. "
                "Noise-cancelling, 40-hour battery life, foldable design. Only $149.99."
            ),
        }
    ),
    "p002": json.dumps(
        {
            "seo_title": "GreenSip EcoBottle 1L – BPA-Free Insulated Water Bottle",
            "seo_description": (
                "Stay hydrated sustainably with the GreenSip EcoBottle. "
                "BPA-free, keeps drinks cold for 24h, leak-proof. Just $24.95."
            ),
        }
    ),
    "p003": json.dumps(
        {
            "seo_title": "BrightSpace LuminaDesk LED Lamp | 5 Brightness Levels",
            "seo_description": (
                "Illuminate your workspace with the BrightSpace LuminaDesk LED Lamp. "
                "Touch control, USB charging port, eye-care mode. $39.00."
            ),
        }
    ),
}


# ---------------------------------------------------------------------------
# Stage functions (pure dataclass transforms)
# ---------------------------------------------------------------------------


def simulate_llm(content: str, source_id: str, stage: str) -> RawLLMResponse:
    """Return a synthetic LLM response using pre-canned JSON strings.

    Args:
        content: The lookup key (source_id).
        source_id: Product identifier for response lookup.
        stage: Pipeline stage name for latency labelling.

    Returns:
        :class:`RawLLMResponse` with simulated content and token counts.
    """
    lookup: dict[str, dict[str, str]] = {
        "extraction": _EXTRACTION_RESPONSES,
        "classification": _CLASSIFICATION_RESPONSES,
        "seo": _SEO_RESPONSES,
    }
    response_text = lookup[stage].get(source_id, '{"error": "unknown product"}')
    return RawLLMResponse(
        content=response_text,
        stop_reason="end_turn",
        input_tokens=80,
        output_tokens=len(response_text) // 4,
        model="claude-opus-4-6-simulated",
        latency_ms=12.0,
        request_id=f"{stage}-{source_id}",
    )


def stage_extract(raw: RawProduct) -> Ok[ExtractedProduct] | Err[str]:
    """Stage 1: extract structured fields from free-form product text.

    Args:
        raw: Raw product with unstructured text.

    Returns:
        Ok wrapping :class:`ExtractedProduct`, or Err with reason.
    """
    config = PreprocessConfig(
        system="Extract product fields from the description as JSON.",
        model="claude-opus-4-6",
        max_tokens=512,
    )
    pp_config = PostprocessConfig(extract_json=True)
    inp = PipelineInput(user_text=raw.raw_text, request_id=raw.source_id)
    _ = preprocess(inp, config)  # shows the real preprocessing contract

    llm_resp = simulate_llm(raw.raw_text, raw.source_id, "extraction")
    output: PipelineOutput = postprocess(llm_resp, pp_config, time.monotonic())

    if not output.has_parsed_output or not isinstance(output.parsed, dict):
        return Err(error=f"extraction failed for {raw.source_id}: no JSON")

    p = output.parsed
    return Ok(
        value=ExtractedProduct(
            name=str(p.get("name", "")),
            brand=str(p.get("brand", "")),
            raw_price=str(p.get("raw_price", "")),
            features=list(p.get("features", [])),
            source_id=raw.source_id,
        )
    )


def stage_classify(extracted: ExtractedProduct) -> Ok[ClassifiedProduct] | Err[str]:
    """Stage 2: classify category, normalise price, determine availability.

    Args:
        extracted: Structured product from stage 1.

    Returns:
        Ok wrapping :class:`ClassifiedProduct`, or Err with reason.
    """
    prompt = f"Classify this product: {extracted.name} at {extracted.raw_price}."
    inp = PipelineInput(user_text=prompt, request_id=extracted.source_id)
    config = PreprocessConfig(system="Return JSON with category, price_usd, in_stock.")
    _ = preprocess(inp, config)

    llm_resp = simulate_llm(prompt, extracted.source_id, "classification")
    output = postprocess(llm_resp, PostprocessConfig(extract_json=True), time.monotonic())

    if not output.has_parsed_output or not isinstance(output.parsed, dict):
        return Err(error=f"classification failed for {extracted.source_id}")

    p = output.parsed
    return Ok(
        value=ClassifiedProduct(
            name=extracted.name,
            brand=extracted.brand,
            category=str(p.get("category", "Uncategorised")),
            price_usd=float(p.get("price_usd", 0.0)),
            in_stock=bool(p.get("in_stock", False)),
            features=extracted.features,
            source_id=extracted.source_id,
        )
    )


def stage_generate_seo(classified: ClassifiedProduct) -> Ok[SEOProduct] | Err[str]:
    """Stage 3: generate SEO title and meta description.

    Args:
        classified: Product with category and price.

    Returns:
        Ok wrapping :class:`SEOProduct`, or Err with reason.
    """
    prompt = f"Write SEO copy for: {classified.name} by {classified.brand}. Price: ${classified.price_usd}."
    inp = PipelineInput(user_text=prompt, request_id=classified.source_id)
    config = PreprocessConfig(system="Return JSON with seo_title and seo_description.")
    _ = preprocess(inp, config)

    llm_resp = simulate_llm(prompt, classified.source_id, "seo")
    output = postprocess(llm_resp, PostprocessConfig(extract_json=True), time.monotonic())

    if not output.has_parsed_output or not isinstance(output.parsed, dict):
        return Err(error=f"SEO generation failed for {classified.source_id}")

    p = output.parsed
    return Ok(
        value=SEOProduct(
            name=classified.name,
            brand=classified.brand,
            category=classified.category,
            price_usd=classified.price_usd,
            in_stock=classified.in_stock,
            seo_title=str(p.get("seo_title", "")),
            seo_description=str(p.get("seo_description", "")),
            source_id=classified.source_id,
        )
    )


# ---------------------------------------------------------------------------
# SQLite storage — raw dataclasses, zero ORM
# ---------------------------------------------------------------------------


@dataclass
class ProductStore:
    """In-memory SQLite store for :class:`SEOProduct` records.

    No ORM. Schema defined as a plain string. Rows mapped to dataclasses
    manually. The connection is owned by this dataclass instance.

    Attributes:
        conn: Open SQLite connection.
        inserted: Count of rows inserted in this session.
    """

    conn: sqlite3.Connection = field(default_factory=lambda: sqlite3.connect(":memory:"))
    inserted: int = 0

    def __post_init__(self) -> None:
        """Create the products table on init."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                source_id   TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                brand       TEXT NOT NULL,
                category    TEXT NOT NULL,
                price_usd   REAL NOT NULL,
                in_stock    INTEGER NOT NULL,
                seo_title   TEXT NOT NULL,
                seo_description TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def upsert(self, product: SEOProduct) -> None:
        """Insert or replace a product row.

        Args:
            product: SEO-enriched product to store.
        """
        row = asdict(product)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO products
                (source_id, name, brand, category, price_usd, in_stock, seo_title, seo_description)
            VALUES
                (:source_id, :name, :brand, :category, :price_usd, :in_stock, :seo_title, :seo_description)
            """,
            row,
        )
        self.conn.commit()
        self.inserted += 1

    def query_by_category(self, category: str) -> list[SEOProduct]:
        """Fetch all products in a category.

        Args:
            category: Category string to filter by.

        Returns:
            List of :class:`SEOProduct` dataclass instances.
        """
        cursor = self.conn.execute(
            "SELECT source_id, name, brand, category, price_usd, in_stock, seo_title, seo_description "
            "FROM products WHERE category = ?",
            (category,),
        )
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        return [
            SEOProduct(**dict(zip(cols, row, strict=True)))  # type: ignore[arg-type]
            for row in rows
        ]

    def all_products(self) -> list[SEOProduct]:
        """Return all stored products.

        Returns:
            List of :class:`SEOProduct` dataclass instances.
        """
        cursor = self.conn.execute(
            "SELECT source_id, name, brand, category, price_usd, in_stock, seo_title, seo_description "
            "FROM products ORDER BY price_usd"
        )
        cols = [d[0] for d in cursor.description]
        return [
            SEOProduct(**dict(zip(cols, row, strict=True)))  # type: ignore[arg-type]
            for row in cursor.fetchall()
        ]


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------


@dataclass
class PipelineStats:
    """Aggregated stats for a pipeline run.

    Attributes:
        total: Input records fed to the pipeline.
        succeeded: Records that reached storage.
        failed: Records that errored in any stage.
        errors: List of error strings for failed records.
    """

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def run_ecommerce_pipeline(raw_products: list[RawProduct], store: ProductStore) -> PipelineStats:
    """Run the full extract → classify → SEO → store pipeline.

    Args:
        raw_products: Input products from a merchant feed.
        store: Destination SQLite store.

    Returns:
        :class:`PipelineStats` with success/failure counts.
    """
    stats = PipelineStats(total=len(raw_products))

    for raw in raw_products:
        # Stage 1 – Extract
        extract_result = stage_extract(raw)
        if extract_result.is_err:
            stats.failed += 1
            stats.errors.append(str(extract_result.error))  # type: ignore[union-attr]
            continue

        # Stage 2 – Classify
        classify_result = stage_classify(extract_result.value)  # type: ignore[union-attr]
        if classify_result.is_err:
            stats.failed += 1
            stats.errors.append(str(classify_result.error))  # type: ignore[union-attr]
            continue

        # Stage 3 – SEO
        seo_result = stage_generate_seo(classify_result.value)  # type: ignore[union-attr]
        if seo_result.is_err:
            stats.failed += 1
            stats.errors.append(str(seo_result.error))  # type: ignore[union-attr]
            continue

        # Stage 4 – Store
        store.upsert(seo_result.value)  # type: ignore[union-attr]
        stats.succeeded += 1

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


SAMPLE_PRODUCTS = [
    RawProduct(
        source_id="p001",
        raw_text=(
            "SonicTech UltraSound Pro 3000 wireless headphones. Industry-leading "
            "noise-cancellation, 40-hour battery. Foldable with USB-C. Only $149.99. In stock."
        ),
    ),
    RawProduct(
        source_id="p002",
        raw_text=(
            "GreenSip EcoBottle 1 litre, BPA-free stainless steel. Keeps drinks cold for 24h. "
            "Leak-proof lid. Dishwasher safe. $24.95. Ships same day."
        ),
    ),
    RawProduct(
        source_id="p003",
        raw_text=(
            "BrightSpace LuminaDesk LED Lamp. 5 brightness levels, USB charging port, "
            "touch control panel, eye-care mode. $39.00. Currently out of stock."
        ),
    ),
]


def main() -> None:
    """Run the e-commerce pipeline and print results."""
    store = ProductStore()
    t0 = time.monotonic()
    stats = run_ecommerce_pipeline(SAMPLE_PRODUCTS, store)
    elapsed_ms = (time.monotonic() - t0) * 1000

    print(f"\n=== E-Commerce Pipeline Results ({elapsed_ms:.1f} ms) ===")
    print(f"Total: {stats.total}  Succeeded: {stats.succeeded}  Failed: {stats.failed}")
    if stats.errors:
        for err in stats.errors:
            print(f"  ERROR: {err}")

    print("\n--- All Products in Store ---")
    for p in store.all_products():
        stock_label = "IN STOCK" if p.in_stock else "OUT OF STOCK"
        print(f"  [{p.category}] {p.name} (${p.price_usd:.2f}) — {stock_label}")
        print(f"    SEO title: {p.seo_title}")

    print("\n--- Electronics Category Query ---")
    electronics = store.query_by_category("Electronics")
    for p in electronics:
        print(f"  {p.name} by {p.brand}")
        print(f"  Meta: {p.seo_description[:80]}...")

    print(f"\nRows stored: {store.inserted}")


if __name__ == "__main__":
    main()
