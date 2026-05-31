# 数据仓库的读写操作

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.retry import retry_async  # [改进] 为 MySQL 读操作添加指数退避重试


class DWMySQLRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_column_types(self, table_name: str) -> dict[str, str]:
        sql = f"show columns from {table_name}"
        # [改进] 读操作加重试，网络抖动不会导致查询失败
        result = await retry_async(self.session.execute, text(sql),
                                   operation_name="MySQL-get_column_types")
        return {row.Field: row.Type for row in result.fetchall()}
    # 给一个表名，查出每个字段叫什么、是什么类型，以字典形式返回。
    # 示例：
    # {
    #     "region_id": "varchar(32)",
    #     "region_name": "varchar(64)",
    #     "level": "int"
    # }

    async def get_column_values(self, table_name: str, column_name: str, limit: int):
        sql = f"select distinct {column_name} from {table_name} limit {limit}"
        # [改进] 读操作加重试
        result = await retry_async(self.session.execute, text(sql),
                                   operation_name="MySQL-get_column_values")
        return result.scalars().fetchall()

    async def get_db_info(self):
        # [改进] 读操作加重试
        result = await retry_async(self.session.execute, text("select version()"),
                                   operation_name="MySQL-get_db_info")
        version = result.scalar()

        dialect = self.session.get_bind().dialect.name

        return {'version': version, 'dialect': dialect}

    async def validate_sql(self, sql):
        # [改进] EXPLAIN 是只读操作，加重试安全
        await retry_async(self.session.execute, text(f"explain {sql}"),
                          operation_name="MySQL-validate_sql")

    async def execute_sql(self, sql):
        # [改进] 注意：execute_sql 是写操作，不加重试——重试可能导致重复执行，造成数据问题
        result = await self.session.execute(text(sql))
        return [dict(row) for row in result.mappings().fetchall()]
