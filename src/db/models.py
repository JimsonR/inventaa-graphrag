"""
SQLAlchemy 2.0 ORM models for Inventaa GraphRAG — Denormalized Tri-Store SQLite schema.

Tables (3):
  products         — Master product catalog with denormalized CSV arrays (categories, features, use_cases)
  product_specs    — Key-value specification entries (FK -> products.sku)
  product_variants — Purchasable variants for color/wattage combinations (FK -> products.sku)

Storage split:
  SQL       — All referential/transactional attributes, pricing, discounts, ratings, images, specs, variants
  Neo4j     — Semantic graph only: HAS_PRODUCT, SUITABLE_FOR, HAS_FEATURE, MADE_BY
  ChromaDB  — Semantic text search chunks (profile descriptions, FAQ text, policies)
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Master Products Table ───────────────────────────────────────────────────────

class Product(Base):
    """
    One row per product.
    Arrays (categories, features, use_cases, color_options, wattage_options) stored as comma-separated text.
    Full specification key-values live in product_specs table.
    Individual SKUs/variants live in product_variants table.
    """
    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String, primary_key=True)                 # Internal ID (e.g. inventaa_product_123)
    sku: Mapped[str] = mapped_column(String, unique=True, index=True)         # Primary SKU (e.g. 12M-2026B)
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)     # Display Name

    # Pricing & Ratings (Authoritative in SQL)
    price_num: Mapped[int] = mapped_column(Integer, default=0, index=True)    # Current sale price in INR
    regular_price: Mapped[Optional[str]] = mapped_column(String)              # MRP before discount
    discount_percentage: Mapped[int] = mapped_column(Integer, default=0)      # % discount
    rating_score: Mapped[float] = mapped_column(Float, default=0.0, index=True) # Star rating (0-5)
    review_count: Mapped[int] = mapped_column(Integer, default=0)             # Review count

    # Media & Links
    image_url: Mapped[Optional[str]] = mapped_column(Text)
    url: Mapped[Optional[str]] = mapped_column(Text)

    # Descriptive Text
    description: Mapped[Optional[str]] = mapped_column(Text)
    feature_descriptions: Mapped[Optional[str]] = mapped_column(Text)
    has_variants: Mapped[bool] = mapped_column(Boolean, default=False)
    wattage: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    tenant: Mapped[str] = mapped_column(String, default="inventaa", index=True)

    # Denormalized Arrays (CSV text for rapid UI filtering and display)
    categories: Mapped[Optional[str]] = mapped_column(Text)                   # "Gate & Pillar Lights,Solar Lights"
    features: Mapped[Optional[str]] = mapped_column(Text)                     # "waterproof,solar-powered,IP65-rated"
    use_cases: Mapped[Optional[str]] = mapped_column(Text)                    # "gate-pillar,garden-pathway"
    color_options: Mapped[Optional[str]] = mapped_column(Text)                # "Cool White,Warm White"
    wattage_options: Mapped[Optional[str]] = mapped_column(Text)              # "12W,18W"

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    specs: Mapped[List["ProductSpec"]] = relationship(
        "ProductSpec", back_populates="product", cascade="all, delete-orphan",
        primaryjoin="Product.sku == ProductSpec.product_sku", foreign_keys="ProductSpec.product_sku"
    )
    variants: Mapped[List["ProductVariant"]] = relationship(
        "ProductVariant", back_populates="product", cascade="all, delete-orphan",
        primaryjoin="Product.sku == ProductVariant.product_sku", foreign_keys="ProductVariant.product_sku"
    )


# ── Key-Value Specifications Table ──────────────────────────────────────────────

class ProductSpec(Base):
    """Normalized key-value specifications for each product (e.g. Wattage: 12W, IP Rating: IP65)."""
    __tablename__ = "product_specs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_sku: Mapped[str] = mapped_column(
        String, ForeignKey("products.sku", ondelete="CASCADE"), index=True
    )
    spec_key: Mapped[str] = mapped_column(String, index=True)                 # e.g. "Wattage", "IP Rating"
    spec_value: Mapped[str] = mapped_column(Text)                             # e.g. "12W", "IP65"

    product: Mapped["Product"] = relationship(
        "Product", back_populates="specs",
        primaryjoin="ProductSpec.product_sku == Product.sku", foreign_keys=[product_sku]
    )


# ── Product Variants Table ──────────────────────────────────────────────────────

class ProductVariant(Base):
    """Individual purchasable variant if a product comes in multiple wattages or colors."""
    __tablename__ = "product_variants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_sku: Mapped[str] = mapped_column(
        String, ForeignKey("products.sku", ondelete="CASCADE"), index=True
    )
    variant_sku: Mapped[Optional[str]] = mapped_column(String, index=True)    # Specific variant SKU if available
    color_option: Mapped[Optional[str]] = mapped_column(String)               # e.g. "Cool White"
    wattage_option: Mapped[Optional[str]] = mapped_column(String)             # e.g. "12W"
    price_num: Mapped[int] = mapped_column(Integer, default=0)                # Price for this specific variant
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)

    product: Mapped["Product"] = relationship(
        "Product", back_populates="variants",
        primaryjoin="ProductVariant.product_sku == Product.sku", foreign_keys=[product_sku]
    )
