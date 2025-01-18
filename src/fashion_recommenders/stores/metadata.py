# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import sqlite3
import os
import json
from pathlib import Path
from typing import List, Optional
from PIL import Image
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import Literal, Tuple

from .. import datatypes
import pathlib
import io
import random


def image_to_bytes(image: Image) -> bytes:
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        return output.getvalue()


class ItemMetadataStore:
    
    def __init__(
        self, 
        database_name: str = 'sqlite',
        table_name: str = 'items',
        base_dir: str = Path.cwd()
    ):
        self.db_path = os.path.join(base_dir, f"{database_name}.db")            
        self.table_name = table_name
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.__create_table()


    def __create_table(self):
        # id autoincrement primary key
        # item_id Candidate key
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            item_id INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            image BLOB NOT NULL,
            description TEXT NOT NULL,
            metadata JSON
        )
        """
        self.conn.execute(query)
        self.conn.commit()
        
        
    def __row_to_item(self, row: Tuple) -> datatypes.FashionItem:
        return datatypes.FashionItem(
            item_id=row[0],
            category=row[1],
            image=Image.open(io.BytesIO(row[2])),
            description=row[3],
            metadata=json.loads(row[4])
        )
        
        
    def get_item(
        self, 
        item_id: int
    ) -> datatypes.FashionItem:
        query = f"""
        SELECT * FROM {self.table_name} 
        WHERE item_id = ?
        """
        cursor = self.conn.execute(query, (item_id,))
        row = cursor.fetchone()
        if row:
            return self.__row_to_item(row)
            
        raise ValueError(
            f"Item with ID {item_id} not found."
        )
        

    def sample_items(
        self, 
        n_samples: int, 
        category: Optional[str] = None
    ) -> List[datatypes.FashionItem]:
        if category is None:
            query = f"""
            SELECT * FROM {self.table_name}
            ORDER BY RANDOM()
            LIMIT ?
            """
            cursor = self.conn.execute(query, (n_samples,))
        else:
            query = f"""
            SELECT * FROM {self.table_name}
            WHERE category = ?
            ORDER BY RANDOM()
            LIMIT ?
            """
            cursor = self.conn.execute(query, (category, n_samples))
        
        return [self.__row_to_item(row) for row in cursor.fetchall()]


    def paginate(
        self, 
        page: int = 1,
        item_per_page: int = 10,
        category: Optional[str] = None,
    ) -> List[datatypes.FashionItem]:
        
        offset = (page - 1) * item_per_page
        
        if category is None:
            query = f"""
            SELECT * FROM {self.table_name}
            LIMIT ? OFFSET ?
            """
            cursor = self.conn.execute(
                query, (item_per_page, offset, )
            )
        else:
            query = f"""
            SELECT * FROM {self.table_name}
            WHERE category = ?
            LIMIT ? OFFSET ?
            """
            cursor = self.conn.execute(
                query, (category, item_per_page, offset, )
            )
            
        return [self.__row_to_item(row) for row in cursor.fetchall()]
        
        
    def total_pages(self, item_per_page: int = 10, category: Optional[str] = None) -> int:
        if category is None:
            query = f"""
            SELECT COUNT(*) FROM {self.table_name}
            """
            
            cursor = self.conn.execute(query)
        else:
            query = f"""
            SELECT COUNT(*) FROM {self.table_name} 
            WHERE category = ?
            """
            
            cursor = self.conn.execute(query, (category,))
        
        return cursor.fetchone()[0] // item_per_page + 1


    def add(
        self, 
        items=List[datatypes.FashionItem]
    ):
        inputs = [
            (item.item_id, item.category, image_to_bytes(item.image), item.description, json.dumps(item.metadata))
            for item in items
        ]
        
        query = f"""
        INSERT OR REPLACE INTO {self.table_name} (item_id, category, image, description, metadata)
        VALUES (?, ?, ?, ?, ?)
        """
            
        self.conn.executemany(query, inputs)
        self.conn.commit()
        
        
    def delete(
        self, 
        item_id: int
    ):
        query = f"""
        DELETE FROM {self.table_name} 
        WHERE item_id = ?
        """
        
        self.conn.execute(query, (item_id,))
        self.conn.commit()


    def __del__(self):
        self.conn.close()