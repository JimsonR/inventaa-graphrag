"""
SQLAlchemy 2.0 ORM models for GraphRAG — Denormalized Tri-Store SQLite schema.

Tables (3):
  products         — Master product catalog with denormalized CSV arrays (categories, features, use_cases)
  product_specs    — Key-value specification entries (FK -> products.sku)
  product_variants — Purchasable variants for option combinations (size/color/wattage/finish) (FK -> products.sku)

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
    Arrays (categories, features, use_cases, options) stored as comma-separated text.
    Full specification key-values live in product_specs table.
    Individual SKUs/variants live in product_variants table.
    """
    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String, primary_key=True)                 # Internal ID (e.g. tenant_product_123)
    sku: Mapped[str] = mapped_column(String, unique=True, index=True)         # Primary SKU (e.g. 12M-2026B)
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)     # Display Name

    # Pricing & Ratings (Authoritative in SQL)
    price_num: Mapped[int] = mapped_column(Integer, default=0, index=True)    # Current sale price (currency defined in tenant config)
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
    primary_option_name: Mapped[Optional[str]] = mapped_column(String)        # Universal primary option name (e.g., "Size", "Wattage")
    primary_options: Mapped[Optional[str]] = mapped_column(Text)              # Universal primary option CSV (e.g., "Small,Medium", "12W,18W")
    wattage: Mapped[Optional[int]] = mapped_column(Integer, index=True)       # Domain attribute (optional)
    tenant: Mapped[str] = mapped_column(String, default="default", index=True)

    # Denormalized Arrays (CSV text for rapid UI filtering and display)
    categories: Mapped[Optional[str]] = mapped_column(Text)                   # e.g., "Dining Tables,Wood Furniture"
    features: Mapped[Optional[str]] = mapped_column(Text)                     # e.g., "solid-oak,extendable"
    use_cases: Mapped[Optional[str]] = mapped_column(Text)                    # e.g., "dining-room,kitchen"
    color_options: Mapped[Optional[str]] = mapped_column(Text)                # e.g., "Oak,Walnut"
    wattage_options: Mapped[Optional[str]] = mapped_column(Text)              # Domain attribute (optional)

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
    """Normalized key-value specifications for each product (e.g. Material: Solid Oak, IP Rating: IP65)."""
    __tablename__ = "product_specs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_sku: Mapped[str] = mapped_column(
        String, ForeignKey("products.sku", ondelete="CASCADE"), index=True
    )
    spec_key: Mapped[str] = mapped_column(String, index=True)                 # e.g. "Material", "Dimensions", "Wattage"
    spec_value: Mapped[str] = mapped_column(Text)                             # e.g. "Solid Oak", "180x90cm", "12W"

    product: Mapped["Product"] = relationship(
        "Product", back_populates="specs",
        primaryjoin="ProductSpec.product_sku == Product.sku", foreign_keys=[product_sku]
    )


# ── Product Variants Table ──────────────────────────────────────────────────────

class ProductVariant(Base):
    """Individual purchasable variant if a product comes in multiple options (size, color, finish, wattage, etc.)."""
    __tablename__ = "product_variants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_sku: Mapped[str] = mapped_column(
        String, ForeignKey("products.sku", ondelete="CASCADE"), index=True
    )
    variant_sku: Mapped[Optional[str]] = mapped_column(String, index=True)    # Specific variant SKU if available
    option_1: Mapped[Optional[str]] = mapped_column(String)                   # Universal primary option value (e.g., size or wattage)
    option_2: Mapped[Optional[str]] = mapped_column(String)                   # Universal secondary option value (e.g., color or finish)
    color_option: Mapped[Optional[str]] = mapped_column(String)               # Domain option attribute (optional)
    wattage_option: Mapped[Optional[str]] = mapped_column(String)             # Domain option attribute (optional)
    price_num: Mapped[int] = mapped_column(Integer, default=0)                # Price for this specific variant
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)

    product: Mapped["Product"] = relationship(
        "Product", back_populates="variants",
        primaryjoin="ProductVariant.product_sku == Product.sku", foreign_keys=[product_sku]
    )
