from typing import List, Callable, Union, Optional
import numpy as np

from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.animate_painters import AnimatePainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

from examples.Machine_learning.SVC._2D.Problems import SVC_2d


class PYProblem:
    """
    Инициализация объекта-задачи
    """

    def __init__(self, dimension: Optional[int] = None, numCriterions: Optional[int] = 1):
        self._functions: List[Callable] = []
        self._lower_bounds: List[float] = []
        self._upper_bounds: List[float] = []
        self._dimension: Optional[int] = dimension
        self._num_crit: Optional[int] = numCriterions

        self.number_of_discrete_variables: Optional[int]
        self.discrete_variable_values: list[list[str]] = []
        self.discrete_variable_names: list[str] = []

    def copy_from_problem(self, problem) -> None:
        """
        Копирует данные из объекта problem в текущий экземпляр PYProblem
    
        Args:
        problem: объект, у которого есть метод calculate и другие атрибуты
        """
        print("copy_from_problem start")
        # Определяем размерность
        if hasattr(problem, 'dimension') and hasattr(problem, "number_of_discrete_variables"):
            self.number_of_discrete_variables = problem.number_of_discrete_variables
            self._dimension = problem.dimension - problem.number_of_discrete_variables
        """
        elif hasattr(problem, '_dimension'):
            self._dimension = problem._dimension
        """
        
        print(f"dimension: {self._dimension}")

        # Копируем границы, если есть
        lower_bounds = None
        upper_bounds = None

        if hasattr(problem, "discrete_variable_values"):
            # Сохраняем в self!
            self.discrete_variable_values = problem.discrete_variable_values
            print(f"DEBUG: discrete_variable_values copied: {self.discrete_variable_values}")
            print(f"DEBUG: type: {type(self.discrete_variable_values)}")
            print(f"DEBUG: length: {len(self.discrete_variable_values) if self.discrete_variable_values else 0}")
        elif hasattr(problem, "kernel_type") and isinstance(problem.kernel_type, dict): # check!!! may be obsolete
            self.discrete_variable_names = list(problem.kernel_type.keys())
            self.discrete_variable_values = list(problem.kernel_type.values())
            print(f"DEBUG: Extracted from kernel_type:")
            print(f"DEBUG:   names: {self.discrete_variable_names}")
            print(f"DEBUG:   values: {self.discrete_variable_values}")
    
        # Детальный вывод каждого дискретного параметра
        if self.discrete_variable_values:
            for i, values in enumerate(self.discrete_variable_values):
                print(f"DEBUG:   discrete param {i}: {values}")
        else:
            print("DEBUG: discrete_variable_values is EMPTY!")

        if hasattr(problem, "discrete_variable_names"):
            self.discrete_variable_names = problem.discrete_variable_names
            print(f"DEBUG: discrete_variable_names copied: {self.discrete_variable_names}")
        else:
            print("DEBUG: discrete_variable_names NOT found in problem")

        if hasattr(problem, 'lower_bound_of_float_variables'):
            lower_bounds = problem.lower_bound_of_float_variables
        elif hasattr(problem, '_lower_bound_of_float_variables'):
            lower_bounds = problem._lower_bound_of_float_variables
    
        if hasattr(problem, 'upper_bound_of_float_variables'):
            upper_bounds = problem.upper_bound_of_float_variables
        elif hasattr(problem, '_upper_bound_of_float_variables'):
            upper_bounds = problem._upper_bound_of_float_variables
    
        if lower_bounds.any() and upper_bounds.any():
            self.set_bounds(lower_bounds, upper_bounds)

        print(f"lower bounds: {self._lower_bounds}; upper bounds: {self._upper_bounds}")
    
        #  Создаём обёртку для метода calculate
        def wrapped_calculate(x):
            """
            Обёртка для вызова метода calculate
        
            Args:
                x: список/массив координат точки
            """

            print(f"[DEBUG] wrapped_calculate called with: {x}")
            print(f"[DEBUG] x type: {type(x)}")

            # Создаём объект point
            if hasattr(problem, 'create_point'):
                print(f"[DEBUG]: started problem.create_point(x)")
                point = problem.create_point(x)
                print(f"[DEBUG]: finished problem.create_point(x)")
            else:
                # Простой объект точки
                print(f"[DEBUG]: create simple point")
                point = type('Point', (), {
                    'float_variables': x,
                    'discrete_variables': x  # или discrete_str_values
                })()    #here modified - check!!!
                print(f"[DEBUG]: finished")
        
            # Создаём объект function_value
            if hasattr(problem, 'create_function_value'):
                function_value = problem.create_function_value()
            else:
                function_value = type('FunctionValue', (), {'value': 0.0})()
        
            # Вызываем метод calculate
            print(f"[DEBUG]: start problem.calculate()")
            result = problem.calculate(point, function_value)   #here!!!
            print(f"[DEBUG]: finished")
        
            # Извлекаем числовое значение
            if hasattr(result, 'value'):
                return result.value
            elif hasattr(function_value, 'value'):
                return function_value.value
            elif isinstance(result, (int, float)):
                return result
            else:
                return float(result)
    
        # Добавляем обёрнутую функцию
        self.add_function(wrapped_calculate, name="calculate_copy")
        print("copy_from_problem finished")


    def add_function(self, func: Callable, name: Optional[str] = None) -> None:
        """
        Добавление одной функции оптимизации

        Args:
            func: функция, принимающая dimension аргументов (или один аргумент-массив)
            name: имя функции (опционально)
        """
        self._functions.append(func)

    def add_functions(self, functions: List[Callable]) -> None:
        """
        Добавление нескольких функций оптимизации
        
        Args:
            functions: список функций
        """
        for func in functions:
            self.add_function(func)

    def set_bounds(self, lower: List[float], upper: List[float]) -> None:
        """
        Установка границ для переменных
        
        Args:
            lower: список нижних границ
            upper: список верхних границ
            
        Raises:
            ValueError: если длины списков не совпадают
        """
        if len(lower) != len(upper):
            raise ValueError(f"Длины lower ({len(lower)}) и upper ({len(upper)}) не совпадают")
        
        if self._dimension is None:
            self._dimension = len(lower)
        elif self._dimension != len(lower):
            raise ValueError(f"Размерность задачи {self._dimension} не совпадает с длиной границ {len(lower)}")
            
        self._lower_bounds = lower.copy()
        self._upper_bounds = upper.copy()
        
    def set_lower_bounds(self, lower: List[float]) -> None:
        """Установка только нижних границ"""
        if self._upper_bounds and len(lower) != len(self._upper_bounds):
            raise ValueError(f"Длина lower ({len(lower)}) не совпадает с upper ({len(self._upper_bounds)})")
        self._lower_bounds = lower.copy()
        if self._dimension is None:
            self._dimension = len(lower)
            
    def set_upper_bounds(self, upper: List[float]) -> None:
        """Установка только верхних границ"""
        if self._lower_bounds and len(upper) != len(self._lower_bounds):
            raise ValueError(f"Длина upper ({len(upper)}) не совпадает с lower ({len(self._lower_bounds)})")
        self._upper_bounds = upper.copy()
        if self._dimension is None:
            self._dimension = len(upper)