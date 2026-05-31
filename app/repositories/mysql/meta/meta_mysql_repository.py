# Meta数据库的读写操作

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache_registry import caches
from app.core.retry import retry_async  # [改进] 为 MySQL 读操作添加指数退避重试
from app.entities.column_info import ColumnInfo
from app.entities.column_metric import ColumnMetric
from app.entities.metric_info import MetricInfo
from app.entities.table_info import TableInfo
from app.models.column_info_mysql import ColumnInfoMySQL
from app.models.table_info_mysql import TableInfoMySQL
from app.repositories.mysql.meta.mappers.column_info_mapper import ColumnInfoMapper
from app.repositories.mysql.meta.mappers.column_metric_mapper import ColumnMetricMapper
from app.repositories.mysql.meta.mappers.metric_info_mapper import MetricInfoMapper
from app.repositories.mysql.meta.mappers.table_info_mapper import TableInfoMapper


class MetaMySQLRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    # [改进] 以下 save_* 为写操作（add_all），不加重试——add_all 只是暂存到 session，
    # 真正写入在事务 commit 时发生，重试 add_all 无意义且可能导致重复数据
    async def save_table_infos(self, table_infos: list[TableInfo]):
        models = [TableInfoMapper.to_model(table_info) for table_info in table_infos]
        self.session.add_all(models)

    async def save_column_infos(self, columns_info: list[ColumnInfo]):
        models = [ColumnInfoMapper.to_model(column_info) for column_info in columns_info]
        self.session.add_all(models)

    async def save_metric_infos(self, metric_infos: list[MetricInfo]):
        self.session.add_all([MetricInfoMapper.to_model(metric_info) for metric_info in metric_infos])

    async def save_column_metrics(self, column_metrics: list[ColumnMetric]):
        self.session.add_all([ColumnMetricMapper.to_model(column_metric) for column_metric in column_metrics])

    async def get_column_info_by_id(self, column_id: str) -> ColumnInfo | None:
        cache_key = f"column:{column_id}"
        cached = caches.meta_mysql.get(cache_key)
        if cached is not None:
            return cached
        # [改进] 读操作加重试
        result: ColumnInfoMySQL | None = await retry_async(self.session.get, ColumnInfoMySQL, column_id,
                                                           operation_name="MySQL-get_column_info_by_id")
        if result:
            entity = ColumnInfoMapper.to_entity(result)
            caches.meta_mysql.set(cache_key, entity)
            return entity
        return None

    async def get_table_info_by_id(self, table_id: str) -> TableInfo | None:
        cache_key = f"table:{table_id}"
        cached = caches.meta_mysql.get(cache_key)
        if cached is not None:
            return cached
        # [改进] 读操作加重试
        result: TableInfoMySQL | None = await retry_async(self.session.get, TableInfoMySQL, table_id,
                                                          operation_name="MySQL-get_table_info_by_id")
        if result:
            entity = TableInfoMapper.to_entity(result)
            caches.meta_mysql.set(cache_key, entity)
            return entity
        return None

    async def get_key_columns_by_table_id(self, table_id: str) -> list[ColumnInfo]:
        cache_key = f"key_cols:{table_id}"
        cached = caches.meta_mysql.get(cache_key)
        if cached is not None:
            return cached
        sql = """
            select *
            from column_info
            where table_id = :table_id
            and role in ('primary_key', 'foreign_key')
        """
        # [改进] 读操作加重试
        result = await retry_async(self.session.execute, text(sql), {"table_id": table_id},
                                   operation_name="MySQL-get_key_columns_by_table_id")
        entities = [ColumnInfo(**row) for row in result.mappings().fetchall()]
        caches.meta_mysql.set(cache_key, entities)
        return entities
