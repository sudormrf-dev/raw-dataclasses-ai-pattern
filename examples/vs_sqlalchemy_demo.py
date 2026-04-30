"""SQLAlchemy ORM vs raw dataclasses + SQL — side-by-side comparison.

Shows the same five operations implemented both ways so an AI agent (or a human)
can compare verbosity, magic and readability without running SQLAlchemy.

The SQLAlchemy side is shown as *code strings* with line counts — the package is
not actually imported, keeping this file dependency-free (stdlib only).

Run::

    python examples/vs_sqlalchemy_demo.py
"""

from __future__ import annotations

import sqlite3
import textwrap
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_non_blank(code: str) -> int:
    """Return the number of non-blank, non-comment lines in a code string."""
    return sum(1 for line in code.splitlines() if line.strip() and not line.strip().startswith("#"))


def _header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def _section(label: str, sqla_code: str, dc_code: str) -> None:
    sqla_lines = _count_non_blank(sqla_code)
    dc_lines = _count_non_blank(dc_code)
    winner = "DC" if dc_lines <= sqla_lines else "SQLAlchemy"
    ratio = sqla_lines / dc_lines if dc_lines else 1.0

    print(f"\n{'─' * 72}")
    print(f"  Operation: {label}")
    print(f"{'─' * 72}")
    print(f"\n  [SQLAlchemy ORM — {sqla_lines} lines]")
    for line in textwrap.dedent(sqla_code).strip().splitlines():
        print(f"    {line}")
    print(f"\n  [Raw Dataclass + SQL — {dc_lines} lines]")
    for line in textwrap.dedent(dc_code).strip().splitlines():
        print(f"    {line}")
    print(f"\n  -> Simpler: {winner}  |  Ratio: {ratio:.1f}x  (SQLAlchemy / DC lines)")


# ---------------------------------------------------------------------------
# Operation 1 — Define model
# ---------------------------------------------------------------------------

SQLA_DEFINE = """\
from sqlalchemy import Column, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class Product(Base):
    __tablename__ = "products"

    id       = Column(Integer, primary_key=True, autoincrement=True)
    name     = Column(String(200), nullable=False)
    brand    = Column(String(100), nullable=False)
    category = Column(String(80), nullable=False)
    price    = Column(Float, nullable=False)
    in_stock = Column(Integer, default=1)
"""

DC_DEFINE = """\
from dataclasses import dataclass

@dataclass
class Product:
    name:     str
    brand:    str
    category: str
    price:    float
    in_stock: bool
    id:       int = 0
"""

# ---------------------------------------------------------------------------
# Operation 2 — Insert
# ---------------------------------------------------------------------------

SQLA_INSERT = """\
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)

with Session(engine) as session:
    p = Product(name="UltraSound Pro", brand="SonicTech",
                category="Electronics", price=149.99, in_stock=1)
    session.add(p)
    session.commit()
"""

DC_INSERT = """\
conn = sqlite3.connect(":memory:")
conn.execute('''
    CREATE TABLE products
    (id INTEGER PRIMARY KEY, name TEXT, brand TEXT,
     category TEXT, price REAL, in_stock INTEGER)
''')
p = Product(name="UltraSound Pro", brand="SonicTech",
            category="Electronics", price=149.99, in_stock=True)
conn.execute(
    "INSERT INTO products (name, brand, category, price, in_stock) VALUES (?,?,?,?,?)",
    (p.name, p.brand, p.category, p.price, int(p.in_stock)),
)
conn.commit()
"""

# ---------------------------------------------------------------------------
# Operation 3 — Query / filter
# ---------------------------------------------------------------------------

SQLA_QUERY = """\
with Session(engine) as session:
    results = (
        session.query(Product)
               .filter(Product.category == "Electronics",
                       Product.price < 200)
               .all()
    )
    for p in results:
        print(p.name, p.price)
"""

DC_QUERY = """\
cursor = conn.execute(
    "SELECT * FROM products WHERE category = ? AND price < ?",
    ("Electronics", 200),
)
cols = [d[0] for d in cursor.description]
results = [Product(**dict(zip(cols, row))) for row in cursor.fetchall()]
for p in results:
    print(p.name, p.price)
"""

# ---------------------------------------------------------------------------
# Operation 4 — Join
# ---------------------------------------------------------------------------

SQLA_JOIN = """\
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

class Review(Base):
    __tablename__ = "reviews"
    id         = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    rating     = Column(Integer)
    product    = relationship("Product", back_populates="reviews")

Product.reviews = relationship("Review", back_populates="product")

with Session(engine) as session:
    rows = (
        session.query(Product, Review)
               .join(Review, Product.id == Review.product_id)
               .filter(Review.rating >= 4)
               .all()
    )
    for product, review in rows:
        print(product.name, review.rating)
"""

DC_JOIN = """\
@dataclass
class Review:
    id:         int
    product_id: int
    rating:     int

cursor = conn.execute('''
    SELECT p.name, r.rating
    FROM products p
    JOIN reviews r ON r.product_id = p.id
    WHERE r.rating >= 4
''')
for name, rating in cursor.fetchall():
    print(name, rating)
"""

# ---------------------------------------------------------------------------
# Operation 5 — Migration
# ---------------------------------------------------------------------------

SQLA_MIGRATION = """\
# Alembic migration (alembic/versions/001_add_sku.py)
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column("products", sa.Column("sku", sa.String(50)))

def downgrade():
    op.drop_column("products", "sku")

# Plus: alembic.ini, env.py, alembic/ directory, alembic revision command
# Total setup: ~6 files, alembic dependency
"""

DC_MIGRATION = """\
# Pure SQL migration — run once
conn.execute("ALTER TABLE products ADD COLUMN sku TEXT DEFAULT ''")
conn.commit()

# Version-tracked in a plain migrations/ folder as numbered .sql files
# No extra dependency, no generated files
"""


# ---------------------------------------------------------------------------
# Live demo: raw DC pipeline actually runs
# ---------------------------------------------------------------------------


@dataclass
class Product:
    """Demo product dataclass (no ORM, no metaclass magic)."""

    name: str
    brand: str
    category: str
    price: float
    in_stock: bool
    id: int = 0


@dataclass
class LiveStore:
    """Thin wrapper around sqlite3 to demonstrate the DC approach works."""

    conn: sqlite3.Connection = field(default_factory=lambda: sqlite3.connect(":memory:"))

    def __post_init__(self) -> None:
        self.conn.execute("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, brand TEXT, category TEXT,
                price REAL, in_stock INTEGER
            )
        """)
        self.conn.execute("""
            CREATE TABLE reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER, rating INTEGER
            )
        """)
        self.conn.commit()

    def insert_product(self, p: Product) -> int:
        cur = self.conn.execute(
            "INSERT INTO products (name, brand, category, price, in_stock) VALUES (?,?,?,?,?)",
            (p.name, p.brand, p.category, p.price, int(p.in_stock)),
        )
        self.conn.commit()
        return cur.lastrowid or 0

    def insert_review(self, product_id: int, rating: int) -> None:
        self.conn.execute(
            "INSERT INTO reviews (product_id, rating) VALUES (?,?)",
            (product_id, rating),
        )
        self.conn.commit()

    def query_electronics_under_200(self) -> list[Product]:
        cur = self.conn.execute(
            "SELECT id, name, brand, category, price, in_stock FROM products "
            "WHERE category = ? AND price < ?",
            ("Electronics", 200),
        )
        cols = [d[0] for d in cur.description]
        return [Product(**dict(zip(cols, row, strict=True))) for row in cur.fetchall()]  # type: ignore[arg-type]

    def top_rated_names(self, min_rating: int = 4) -> list[str]:
        cur = self.conn.execute(
            "SELECT p.name FROM products p JOIN reviews r ON r.product_id = p.id WHERE r.rating >= ?",
            (min_rating,),
        )
        return [row[0] for row in cur.fetchall()]


def run_live_demo() -> None:
    """Execute the DC-based operations and print real results."""
    store = LiveStore()
    pid = store.insert_product(
        Product(
            name="UltraSound Pro 3000",
            brand="SonicTech",
            category="Electronics",
            price=149.99,
            in_stock=True,
        )
    )
    store.insert_product(
        Product(
            name="EcoBottle 1L",
            brand="GreenSip",
            category="Sports & Outdoors",
            price=24.95,
            in_stock=True,
        )
    )
    store.insert_review(product_id=pid, rating=5)
    store.insert_review(product_id=pid, rating=4)

    electronics = store.query_electronics_under_200()
    top = store.top_rated_names(min_rating=4)

    print("\n--- Live DC demo results ---")
    print(f"  Electronics <$200: {[p.name for p in electronics]}")
    print(f"  Top-rated products: {top}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


@dataclass
class ComparisonRow:
    """One row in the comparison summary table."""

    operation: str
    sqla_lines: int
    dc_lines: int

    @property
    def ratio(self) -> float:
        return self.sqla_lines / self.dc_lines if self.dc_lines else 1.0

    @property
    def simpler(self) -> str:
        return "DC" if self.dc_lines <= self.sqla_lines else "SQLAlchemy"


def _build_summary(pairs: list[tuple[str, str, str]]) -> list[ComparisonRow]:
    return [
        ComparisonRow(
            operation=label,
            sqla_lines=_count_non_blank(sqla),
            dc_lines=_count_non_blank(dc),
        )
        for label, sqla, dc in pairs
    ]


def main() -> None:
    """Print the full comparison and conclusions."""
    _header("SQLAlchemy ORM vs Raw Dataclasses + SQL")

    operations: list[tuple[str, str, str]] = [
        ("1. Define model", SQLA_DEFINE, DC_DEFINE),
        ("2. Insert", SQLA_INSERT, DC_INSERT),
        ("3. Query / filter", SQLA_QUERY, DC_QUERY),
        ("4. Join", SQLA_JOIN, DC_JOIN),
        ("5. Migration", SQLA_MIGRATION, DC_MIGRATION),
    ]

    for label, sqla, dc in operations:
        _section(label, sqla, dc)

    # Summary table
    rows = _build_summary(operations)
    total_sqla = sum(r.sqla_lines for r in rows)
    total_dc = sum(r.dc_lines for r in rows)
    overall_ratio = total_sqla / total_dc if total_dc else 1.0

    print(f"\n{'=' * 72}")
    print("  Summary")
    print(f"{'=' * 72}")
    print(f"  {'Operation':<30} {'SQLAlchemy':>12} {'Raw DC':>8} {'Ratio':>7} {'Winner':>10}")
    print(f"  {'-' * 68}")
    for r in rows:
        print(
            f"  {r.operation:<30} {r.sqla_lines:>12} {r.dc_lines:>8} {r.ratio:>6.1f}x {r.simpler:>10}"
        )
    print(f"  {'-' * 68}")
    print(f"  {'TOTAL':<30} {total_sqla:>12} {total_dc:>8} {overall_ratio:>6.1f}x {'DC':>10}")

    print(f"""
  Conclusion
  ----------
  Raw dataclasses + SQL use ~{overall_ratio:.1f}x fewer lines for the same operations.

  Why this matters for AI agents:
    - No ORM metaclass magic — the agent reads actual Python, not __table_args__
    - SQL is deterministic: the agent can predict the exact query executed
    - No lazy-loading surprises (N+1 queries hidden behind attribute access)
    - Zero extra dependencies → smaller context, fewer hallucination sources
    - Migrations are plain .sql files, not Alembic revision objects
    - Dataclasses are transparent: asdict() / fields() work without ORM mixins
""")

    run_live_demo()


if __name__ == "__main__":
    main()
