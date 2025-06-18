"""Unit tests for labeler module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import col
import time

from madmatcher_tools._internal.labeler import (
    Labeler, GoldLabeler, DelayedGoldLabeler, CLILabeler, CustomLabeler
)


@pytest.mark.unit
class TestLabeler:
    """Test Labeler abstract base class."""

    def test_labeler_abstract_class(self):
        """Test that Labeler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Labeler()


@pytest.mark.unit
class TestGoldLabeler:
    """Test GoldLabeler functionality."""

    def test_gold_labeler_creation(self):
        """Test creating GoldLabeler."""
        gold_data = pd.DataFrame({
            'id1': [1, 2, 3],
            'id2': [101, 102, 103]
        })
        
        labeler = GoldLabeler(gold_data)
        
        assert isinstance(labeler, Labeler)
        assert (1, 101) in labeler._gold
        assert (2, 102) in labeler._gold
        assert (3, 103) in labeler._gold

    def test_gold_labeler_call_match(self):
        """Test GoldLabeler with matching pairs."""
        gold_data = pd.DataFrame({
            'id1': [1, 2, 3],
            'id2': [101, 102, 103]
        })
        
        labeler = GoldLabeler(gold_data)
        
        # Test matching pairs
        assert labeler(1, 101) == 1.0
        assert labeler(2, 102) == 1.0
        assert labeler(3, 103) == 1.0

    def test_gold_labeler_call_no_match(self):
        """Test GoldLabeler with non-matching pairs."""
        gold_data = pd.DataFrame({
            'id1': [1, 2, 3],
            'id2': [101, 102, 103]
        })
        
        labeler = GoldLabeler(gold_data)
        
        # Test non-matching pairs
        assert labeler(1, 102) == 0.0
        assert labeler(4, 104) == 0.0
        assert labeler(2, 101) == 0.0

    def test_gold_labeler_empty_gold(self):
        """Test GoldLabeler with empty gold data."""
        gold_data = pd.DataFrame(columns=['id1', 'id2'])
        
        labeler = GoldLabeler(gold_data)
        
        # All pairs should be non-matches
        assert labeler(1, 101) == 0.0
        assert labeler(2, 102) == 0.0

    def test_gold_labeler_duplicate_pairs(self):
        """Test GoldLabeler with duplicate pairs in gold data."""
        gold_data = pd.DataFrame({
            'id1': [1, 1, 2],
            'id2': [101, 101, 102]  # Duplicate pair (1, 101)
        })
        
        labeler = GoldLabeler(gold_data)
        
        # Should still work correctly
        assert labeler(1, 101) == 1.0
        assert labeler(2, 102) == 1.0
        assert labeler(1, 102) == 0.0


@pytest.mark.unit
class TestDelayedGoldLabeler:
    """Test DelayedGoldLabeler functionality."""

    def test_delayed_gold_labeler_creation(self):
        """Test creating DelayedGoldLabeler."""
        gold_data = pd.DataFrame({
            'id1': [1, 2, 3],
            'id2': [101, 102, 103]
        })
        
        labeler = DelayedGoldLabeler(gold_data, delay_secs=0.1)
        
        assert isinstance(labeler, Labeler)
        # The _gold should now be a set of tuples
        expected_gold = {(1, 101), (2, 102), (3, 103)}
        assert labeler._gold == expected_gold
        assert labeler._delay_secs == 0.1

    def test_delayed_gold_labeler_call_match(self):
        """Test DelayedGoldLabeler with matching pairs."""
        gold_data = pd.DataFrame({
            'id1': [1, 2, 3],
            'id2': [101, 102, 103]
        })

        labeler = DelayedGoldLabeler(gold_data, delay_secs=0.01)

        # Test matching pairs - these should be in the gold_data DataFrame rows
        # The DelayedGoldLabeler now converts DataFrame to set of tuples
        assert labeler(1, 101) == 1.0
        assert labeler(2, 102) == 1.0
        assert labeler(3, 103) == 1.0

        # Non-matching pairs should return 0.0
        assert labeler(1, 102) == 0.0
        assert labeler(2, 103) == 0.0

    def test_delayed_gold_labeler_call_no_match(self):
        """Test DelayedGoldLabeler with non-matching pairs."""
        gold_data = pd.DataFrame({
            'id1': [1, 2, 3],
            'id2': [101, 102, 103]
        })
        
        labeler = DelayedGoldLabeler(gold_data, delay_secs=0.01)
        
        # Test non-matching pairs
        assert labeler(1, 102) == 0.0
        assert labeler(4, 104) == 0.0

    def test_delayed_gold_labeler_zero_delay(self):
        """Test DelayedGoldLabeler with zero delay."""
        gold_data = pd.DataFrame({
            'id1': [1],
            'id2': [101]
        })

        labeler = DelayedGoldLabeler(gold_data, delay_secs=0.0)

        # Should work without delay
        assert labeler(1, 101) == 1.0

    def test_delayed_gold_labeler_dataframe_membership(self):
        """Test how DelayedGoldLabeler checks DataFrame membership."""
        gold_data = pd.DataFrame({
            'id1': [1, 2, 3],
            'id2': [101, 102, 103]
        })
        
        # Test what the actual behavior should be
        # The DelayedGoldLabeler now converts DataFrame to set of tuples
        labeler = DelayedGoldLabeler(gold_data, delay_secs=0.0)
        
        # Test the actual labeler behavior
        assert labeler(1, 101) == 1.0
        assert labeler(2, 102) == 1.0
        assert labeler(3, 103) == 1.0
        # Non-matching pairs should return 0.0
        assert labeler(1, 102) == 0.0
        assert labeler(4, 104) == 0.0


@pytest.mark.unit
class TestCLILabeler:
    """Test CLILabeler functionality."""

    def test_cli_labeler_creation(self):
        """Test creating CLILabeler."""
        a_df = pd.DataFrame({
            '_id': [1, 2],
            'name': ['John', 'Jane'],
            'age': [25, 30]
        })
        
        b_df = pd.DataFrame({
            '_id': [101, 102],
            'name': ['John Smith', 'Jane Doe'],
            'age': [25, 30]
        })
        
        labeler = CLILabeler(a_df, b_df)
        
        assert isinstance(labeler, Labeler)
        assert labeler._a_df.equals(a_df)
        assert labeler._b_df.equals(b_df)
        assert labeler._id_col == '_id'
        assert labeler._all_fields is None
        assert labeler._current_fields is None

    def test_cli_labeler_creation_custom_id_col(self):
        """Test creating CLILabeler with custom id column."""
        a_df = pd.DataFrame({
            'custom_id': [1, 2],
            'name': ['John', 'Jane']
        })
        
        b_df = pd.DataFrame({
            'custom_id': [101, 102],
            'name': ['John Smith', 'Jane Doe']
        })
        
        labeler = CLILabeler(a_df, b_df, id_col='custom_id')
        
        assert labeler._id_col == 'custom_id'

    def test_cli_labeler_get_row_pandas(self):
        """Test _get_row method with pandas DataFrame."""
        a_df = pd.DataFrame({
            '_id': [1, 2],
            'name': ['John', 'Jane'],
            'age': [25, 30]
        })
        
        b_df = pd.DataFrame({
            '_id': [101, 102],
            'name': ['John Smith', 'Jane Doe'],
            'age': [25, 30]
        })
        
        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age'])

        # Test getting existing row
        row = labeler._get_row(a_df, 1)
        assert row['_id'] == 1
        assert row['name'] == 'John'
        assert row['age'] == 25
        
        # Test getting non-existing row
        with pytest.raises(KeyError):
            labeler._get_row(a_df, 999)

    def test_cli_labeler_get_row_spark(self, spark_session):
        """Test _get_row method with Spark DataFrame."""
        a_df = spark_session.createDataFrame([
            (1, 'John', 25),
            (2, 'Jane', 30)
        ], ['_id', 'name', 'age'])
        
        b_df = spark_session.createDataFrame([
            (101, 'John Smith', 25),
            (102, 'Jane Doe', 30)
        ], ['_id', 'name', 'age'])
        
        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age'])

        # Test getting existing row
        row = labeler._get_row(a_df, 1)
        assert row['_id'] == 1
        assert row['name'] == 'John'
        assert row['age'] == 25
        
        # Test getting non-existing row
        with pytest.raises(KeyError):
            labeler._get_row(a_df, 999)

    def test_cli_labeler_print_dict(self):
        """Test _print_dict static method."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John']
        })
        
        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith']
        })
        
        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age'])

        # Test with a simple dict
        test_dict = {'name': 'John', 'age': 25}
        
        # This should not raise an exception
        labeler._print_dict(test_dict)

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_call_basic(self, mock_print, mock_input):
        """Test CLILabeler __call__ method with basic input."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John'],
            'age': [25]
        })
        
        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith'],
            'age': [25]
        })
        
        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age'])

        # Mock user input
        mock_input.return_value = 'y'
        
        result = labeler(1, 101)
        
        assert result == 1.0
        assert mock_print.called
        assert mock_input.called

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_call_no_match(self, mock_print, mock_input):
        """Test CLILabeler __call__ method with no match input."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John'],
            'age': [25]
        })
        
        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith'],
            'age': [25]
        })
        
        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age'])

        # Mock user input
        mock_input.return_value = 'n'
        
        result = labeler(1, 101)
        
        assert result == 0.0

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_call_unsure(self, mock_print, mock_input):
        """Test CLILabeler __call__ method with unsure input."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John'],
            'age': [25]
        })
        
        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith'],
            'age': [25]
        })
        
        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age'])

        # Mock user input
        mock_input.return_value = 'u'
        
        result = labeler(1, 101)
        
        assert result == 2.0

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_call_stop(self, mock_print, mock_input):
        """Test CLILabeler __call__ method with stop input."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John'],
            'age': [25]
        })
        
        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith'],
            'age': [25]
        })
        
        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age'])

        # Mock user input sequence: stop, then confirm
        mock_input.side_effect = ['s', 'y']
        
        result = labeler(1, 101)
        
        assert result == -1.0

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_call_help(self, mock_print, mock_input):
        """Test CLILabeler __call__ method with help input."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John'],
            'age': [25]
        })
        
        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith'],
            'age': [25]
        })
        
        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age'])

        # Mock user input sequence: help, then yes
        mock_input.side_effect = ['h', 'e', 'y']  # help, exit help, yes
        
        result = labeler(1, 101)
        
        assert result == 1.0
        # Should have called print more times due to help
        assert mock_print.call_count > 1

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_call_invalid_input(self, mock_print, mock_input):
        """Test CLILabeler __call__ method with invalid input."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John'],
            'age': [25]
        })
        
        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith'],
            'age': [25]
        })
        
        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age'])

        # Mock user input sequence: invalid, then yes
        mock_input.side_effect = ['invalid', 'y']
        
        result = labeler(1, 101)
        
        assert result == 1.0

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_help_interactive(self, mock_print, mock_input):
        """Test _help_interactive method."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John'],
            'age': [25]
        })
        
        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith'],
            'age': [25]
        })
        
        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age'])

        # Mock user input to exit help
        mock_input.return_value = 'e'
        
        row1 = {'name': 'John', 'age': 25}
        row2 = {'name': 'John Smith', 'age': 25}
        
        # This should not raise an exception
        labeler._help_interactive(row1, row2)

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_help_interactive_add_remove(self, mock_print, mock_input):
        """Test _help_interactive method with add/remove commands."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John'],
            'age': [25],
            'city': ['NYC']
        })

        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith'],
            'age': [25],
            'city': ['NYC']
        })

        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age', 'city']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age', 'city'])

        # Mock user input sequence: add field, remove field, exit
        mock_input.side_effect = ['a', '1', 'r', '1', 'e']

        row1 = {'name': 'John', 'age': 25, 'city': 'NYC'}
        row2 = {'name': 'John Smith', 'age': 25, 'city': 'NYC'}

        # This should not raise an exception
        labeler._help_interactive(row1, row2)

    @patch('builtins.input')
    @patch('builtins.print')
    def test_cli_labeler_help_interactive_invalid_commands(self, mock_print, mock_input):
        """Test _help_interactive method with invalid commands."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John'],
            'age': [25]
        })

        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith'],
            'age': [25]
        })

        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age'])

        # Mock user input sequence: invalid command, then exit
        mock_input.side_effect = ['invalid', 'e']

        row1 = {'name': 'John', 'age': 25}
        row2 = {'name': 'John Smith', 'age': 25}

        # This should not raise an exception
        labeler._help_interactive(row1, row2)

    @patch('shutil.get_terminal_size')
    @patch('builtins.print')
    def test_cli_labeler_print_row(self, mock_print, mock_terminal_size):
        """Test _print_row method."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John'],
            'age': [25]
        })
        
        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith'],
            'age': [25]
        })
        
        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age'])

        # Mock terminal size with proper object
        mock_terminal_size.return_value = Mock(columns=120, lines=20)
        
        row1 = {'name': 'John', 'age': 25}
        row2 = {'name': 'John Smith', 'age': 25}
        fields = ['name', 'age']
        
        # This should not raise an exception
        labeler._print_row(row1, row2, fields)

    @patch('shutil.get_terminal_size')
    @patch('builtins.print')
    def test_cli_labeler_print_row_long_content(self, mock_print, mock_terminal_size):
        """Test _print_row method with long content."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John'],
            'age': [25]
        })
        
        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith'],
            'age': [25]
        })
        
        labeler = CLILabeler(a_df, b_df)
        
        # Initialize _all_fields and _current_fields if not already set
        if labeler._all_fields is None:
            labeler._all_fields = ['name', 'age']
        if labeler._current_fields is None:
            labeler._current_fields = set(['name', 'age'])

        # Mock terminal size with proper object
        mock_terminal_size.return_value = Mock(columns=120, lines=20)
        
        row1 = {'name': 'John' * 50, 'age': 25}  # Very long name
        row2 = {'name': 'John Smith' * 50, 'age': 25}  # Very long name
        fields = ['name', 'age']
        
        # This should not raise an exception
        labeler._print_row(row1, row2, fields)


@pytest.mark.unit
class TestCustomLabeler:
    """Test CustomLabeler functionality."""

    def test_custom_labeler_creation(self):
        """Test creating CustomLabeler."""
        a_df = pd.DataFrame({
            '_id': [1, 2],
            'name': ['John', 'Jane'],
            'age': [25, 30]
        })
        
        b_df = pd.DataFrame({
            '_id': [101, 102],
            'name': ['John Smith', 'Jane Doe'],
            'age': [25, 30]
        })
        
        # CustomLabeler is abstract, so we need to create a concrete implementation
        class TestCustomLabeler(CustomLabeler):
            def label_pair(self, row1, row2):
                return 1.0 if row1['name'] == row2['name'] else 0.0
        
        labeler = TestCustomLabeler(a_df, b_df)
        
        assert isinstance(labeler, Labeler)
        assert labeler._a_df.equals(a_df)
        assert labeler._b_df.equals(b_df)
        assert labeler._id_col == '_id'

    def test_custom_labeler_creation_custom_id_col(self):
        """Test creating CustomLabeler with custom id column."""
        a_df = pd.DataFrame({
            'custom_id': [1, 2],
            'name': ['John', 'Jane']
        })
        
        b_df = pd.DataFrame({
            'custom_id': [101, 102],
            'name': ['John Smith', 'Jane Doe']
        })
        
        class TestCustomLabeler(CustomLabeler):
            def label_pair(self, row1, row2):
                return 1.0 if row1['name'] == row2['name'] else 0.0
        
        labeler = TestCustomLabeler(a_df, b_df, id_col='custom_id')
        
        assert labeler._id_col == 'custom_id'

    def test_custom_labeler_call(self):
        """Test CustomLabeler __call__ method."""
        a_df = pd.DataFrame({
            '_id': [1, 2],
            'name': ['John', 'Jane'],
            'age': [25, 30]
        })
        
        b_df = pd.DataFrame({
            '_id': [101, 102],
            'name': ['John', 'Jane Doe'],
            'age': [25, 30]
        })
        
        class TestCustomLabeler(CustomLabeler):
            def label_pair(self, row1, row2):
                return 1.0 if row1['name'] == row2['name'] else 0.0
        
        labeler = TestCustomLabeler(a_df, b_df)
        
        # Test matching names
        assert labeler(1, 101) == 1.0  # Both have 'John'
        
        # Test non-matching names
        assert labeler(2, 102) == 0.0  # 'Jane' vs 'Jane Doe'

    def test_custom_labeler_get_row_pandas(self):
        """Test _get_row method with pandas DataFrame."""
        a_df = pd.DataFrame({
            '_id': [1, 2],
            'name': ['John', 'Jane'],
            'age': [25, 30]
        })
        
        b_df = pd.DataFrame({
            '_id': [101, 102],
            'name': ['John Smith', 'Jane Doe'],
            'age': [25, 30]
        })
        
        class TestCustomLabeler(CustomLabeler):
            def label_pair(self, row1, row2):
                return 1.0
        
        labeler = TestCustomLabeler(a_df, b_df)
        
        # Test getting existing row
        row = labeler._get_row(a_df, 1)
        assert row['_id'] == 1
        assert row['name'] == 'John'
        assert row['age'] == 25
        
        # Test getting non-existing row
        with pytest.raises(KeyError):
            labeler._get_row(a_df, 999)

    def test_custom_labeler_get_row_spark(self, spark_session):
        """Test _get_row method with Spark DataFrame."""
        a_df = spark_session.createDataFrame([
            (1, 'John', 25),
            (2, 'Jane', 30)
        ], ['_id', 'name', 'age'])
        
        b_df = spark_session.createDataFrame([
            (101, 'John Smith', 25),
            (102, 'Jane Doe', 30)
        ], ['_id', 'name', 'age'])
        
        class TestCustomLabeler(CustomLabeler):
            def label_pair(self, row1, row2):
                return 1.0
        
        labeler = TestCustomLabeler(a_df, b_df)
        
        # Test getting existing row
        row = labeler._get_row(a_df, 1)
        assert row['_id'] == 1
        assert row['name'] == 'John'
        assert row['age'] == 25
        
        # Test getting non-existing row
        with pytest.raises(KeyError):
            labeler._get_row(a_df, 999)

    def test_custom_labeler_print_dict(self):
        """Test _print_dict static method."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John']
        })
        
        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith']
        })
        
        class TestCustomLabeler(CustomLabeler):
            def label_pair(self, row1, row2):
                return 1.0
        
        labeler = TestCustomLabeler(a_df, b_df)
        
        # Test with a simple dict
        test_dict = {'name': 'John', 'age': 25}
        
        # This should not raise an exception
        labeler._print_dict(test_dict)

    def test_custom_labeler_abstract_method(self):
        """Test that CustomLabeler cannot be instantiated without implementing label_pair."""
        a_df = pd.DataFrame({
            '_id': [1],
            'name': ['John']
        })
        
        b_df = pd.DataFrame({
            '_id': [101],
            'name': ['John Smith']
        })
        
        with pytest.raises(TypeError):
            CustomLabeler(a_df, b_df)


@pytest.mark.unit
class TestLabelerIntegration:
    """Test integration scenarios for labelers."""

    def test_labeler_with_spark_dataframes(self, spark_session):
        """Test labelers with Spark DataFrames."""
        a_df = spark_session.createDataFrame([
            (1, 'John', 25),
            (2, 'Jane', 30)
        ], ['_id', 'name', 'age'])
        
        b_df = spark_session.createDataFrame([
            (101, 'John Smith', 25),
            (102, 'Jane Doe', 30)
        ], ['_id', 'name', 'age'])
        
        # Test GoldLabeler with Spark DataFrames
        gold_data = spark_session.createDataFrame([
            (1, 101),
            (2, 102)
        ], ['id1', 'id2'])
        
        labeler = GoldLabeler(gold_data.toPandas())
        
        assert labeler(1, 101) == 1.0
        assert labeler(2, 102) == 1.0
        assert labeler(1, 102) == 0.0

    def test_labeler_error_handling(self):
        """Test labeler error handling scenarios."""
        # Test GoldLabeler with missing columns
        with pytest.raises(KeyError):
            gold_data = pd.DataFrame({
                'wrong_col1': [1, 2],
                'wrong_col2': [101, 102]
            })
            GoldLabeler(gold_data)

    def test_labeler_with_empty_dataframes(self):
        """Test labelers with empty DataFrames."""
        empty_df = pd.DataFrame(columns=['id1', 'id2'])
        
        labeler = GoldLabeler(empty_df)
        
        # Should handle empty data gracefully
        assert labeler(1, 101) == 0.0
        assert labeler(2, 102) == 0.0

    def test_labeler_with_none_values(self):
        """Test labelers with None values in data."""
        gold_data = pd.DataFrame({
            'id1': [1, None, 3],
            'id2': [101, 102, None]
        })
        
        labeler = GoldLabeler(gold_data)
        
        # Should handle None values
        assert labeler(1, 101) == 1.0
        assert labeler(None, 102) == 0.0  # None values should not match
        assert labeler(3, None) == 0.0
