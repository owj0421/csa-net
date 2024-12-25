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
from typing import Literal
from ..utils import elements


class BaseItemLoader(ABC):
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def get_item(self, item_id: str) -> elements.Item:
        pass
    
    @abstractmethod
    def get_category(self, item_id: str) -> str:
        pass
    
    @abstractmethod
    def sample_items(self, n_samples: int) -> List[str]:
        pass
    
    @abstractmethod
    def sample_items_by_category(self, category: str, n_samples: int) -> List[str]:
        pass
    
    def sample_items_by_id(self, item_id: str, n_samples: int) -> List[str]:
        return self.sample_items_by_category(
            category=self.get_category(item_id), 
            n_samples=n_samples
        )
    
    @abstractmethod
    def paginate_items(
        self, 
        page: int = 1,
        item_per_page: int = 10,
        category: Optional[str] = None
    ) -> List[elements.Item]:
        pass
    
    @abstractmethod
    def total_pages(self, item_per_page: int = 10, category: Optional[str] = None) -> int:
        pass
    
    @abstractmethod
    def add(self, item_id, description, category):
        pass
    
    @abstractmethod
    def delete(self, item_id):
        pass
    
    def __call__(self, item_id: str) -> elements.Item:
        return self.get_item(item_id)
    

class SQLiteItemLoader(BaseItemLoader):
    
    def __init__(
        self, 
        db_dir: str,
        # image_dir: str,
    ):
        self.db_path = db_dir + '/items.db'
        if not Path(db_dir).exists():
            os.makedirs(db_dir, exist_ok=True)
            print(
                f"Directory {db_dir} created."
            )
            
        # self.image_dir = image_dir
        # if not Path(image_dir).exists():
        #     raise ValueError(
        #         f"Image directory {image_dir} not found."
        #     )
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS items (
            item_id INTEGER PRIMARY KEY,
            image_path TEXT,
            description TEXT,
            category TEXT
        )
        """
        
        self.conn.execute(query)
        self.conn.commit()
        
    # def _get_image(self, item_id: int) -> Image.Image:
    #     path = Path(self.image_dir) / f"{item_id}.jpg"
    #     if path.exists():
    #         return Image.open(path)
        
    #     raise ValueError(
    #         f"Image for item ID {item_id} not found."
    #     )

    def get_item(self, item_id: int) -> elements.Item:
        query = """
        SELECT * FROM items 
        WHERE item_id = ?
        """
        
        cursor = self.conn.execute(query, (item_id,))
        row = cursor.fetchone()
        if row:
            return elements.Item(
                item_id=row[0],
                image=Image.open(row[1]),
                description=row[2],
                category=row[3],
            )
            
        raise ValueError(
            f"Item with ID {item_id} not found."
        )

    def get_category(self, item_id: int) -> str:
        query = """
        SELECT category FROM items 
        WHERE item_id = ?
        """
        
        cursor = self.conn.execute(query, (item_id,))
        row = cursor.fetchone()
        if row:
            return row[0]
        
        raise ValueError(
            f"Category for item ID {item_id} not found."
        )

    def sample_items(
        self, n_samples: int,
        return_type: Literal['item', 'id'] = 'item'
    ) -> List[str]:
        query = """
        SELECT item_id FROM items 
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        cursor = self.conn.execute(query, (n_samples,))
        if return_type == 'id':
            return [row[0] for row in cursor.fetchall()]
        elif return_type == 'item':
            return [self.get_item(row[0]) for row in cursor.fetchall()]

    def sample_items_by_category(
        self, category: str, n_samples: int,
        return_type: Literal['item', 'id'] = 'item'
    ) -> List[str]:
        query = """
        SELECT item_id FROM items 
        WHERE category = ? 
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        cursor = self.conn.execute(query, (category, n_samples))
        if return_type == 'id':
            return [row[0] for row in cursor.fetchall()]
        elif return_type == 'item':
            return [self.get_item(row[0]) for row in cursor.fetchall()]
    
    def paginate_items(
        self, 
        page: int = 1,
        item_per_page: int = 10,
        category: Optional[str] = None,
        return_type: Literal['item', 'id'] = 'item'
    ) -> List[elements.Item]:
        if category is None:
            query = """
            SELECT * FROM items 
            LIMIT ? OFFSET ?
            """
            
            cursor = self.conn.execute(
                query, 
                (item_per_page, (page - 1) * item_per_page)
            )
        else:
            query = """
            SELECT * FROM items 
            WHERE category = ?
            LIMIT ? OFFSET ?
            """
            
            cursor = self.conn.execute(
                query, 
                (category, item_per_page, (page - 1) * item_per_page)
            )
            
            
        if return_type == 'id':
            return [row[0] for row in cursor.fetchall()]
            
        elif return_type == 'item':
            return [self.get_item(row[0]) for row in cursor.fetchall()]
        
    def total_pages(self, item_per_page: int = 10, category: Optional[str] = None) -> int:
        if category is None:
            query = """
            SELECT COUNT(*) FROM items
            """
            
            cursor = self.conn.execute(query)
        else:
            query = """
            SELECT COUNT(*) FROM items 
            WHERE category = ?
            """
            
            cursor = self.conn.execute(query, (category,))
        
        return cursor.fetchone()[0] // item_per_page + 1

    def add(self, items=List[elements.Item]):
        inputs = [
            (item.item_id, item.image_path,  item.description, item.category)
            for item in items
        ]
        
        query = """
        INSERT OR REPLACE INTO items (item_id, image_path, description, category)
        VALUES (?, ?, ?, ?)
        """
            
        self.conn.executemany(query, inputs)
        self.conn.commit()
        
    def delete(self, item_id: int):
        query = """
        DELETE FROM items 
        WHERE item_id = ?
        """
        
        self.conn.execute(query, (item_id,))
        self.conn.commit()

    def __del__(self):
        self.conn.close()
        
    def __call__(self, item_id: int) -> elements.Item:
        return self.get_item(item_id)