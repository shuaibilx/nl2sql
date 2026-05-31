from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

# ORM实体类（对应META数据库的表）
# 对应 column_metric 关系表
#

class ColumnMetricMySQL(Base):
    __tablename__ = "column_metric"

    column_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        comment="列编号"
    )
    metric_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        comment="指标编号"
    )
