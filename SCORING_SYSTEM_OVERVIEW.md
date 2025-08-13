# IMIS Scheduler Scoring System

## Overview

The new scoring system has been implemented to optimize provider scheduling based on work-life balance principles and clinical requirements. This system replaces the previous ad-hoc assignment methods with a comprehensive scoring algorithm that considers multiple factors.

## Key Features

### 1. Block-Based Scheduling
- **Optimal block size**: 3-7 consecutive shifts
- **Block consistency**: Same shift types within blocks (no mixing day/night shifts)
- **Rest periods**: Minimum 2+ days between blocks
- **Block penalties**: Strong penalties for standalone days and very short blocks

### 2. Provider Type Specialization
- **APP Providers**: Only APP shifts, focus on weekend coverage
- **Nocturnists**: Only night shifts (N12, NB), optimized for night blocks
- **Seniors**: Only rounding shifts (R12), shorter blocks preferred
- **Regular Providers**: All shifts except APP, balanced day/night ratios

### 3. Work-Life Balance Optimization
- **Timing preferences**: Front-loaded vs back-loaded schedule support
- **Weekend distribution**: Fair allocation of weekend shifts
- **Day/night ratios**: Respects provider preferences for day vs night work
- **Vacation avoidance**: Prevents assignments during unavailable periods

### 4. Clinical Safety
- **No dangerous transitions**: Prevents day-to-night shifts without adequate rest
- **Rest requirements**: Enforces minimum rest days between shift blocks
- **Workload limits**: Prevents provider overwork through shift count limits
- **Shift type preferences**: Respects provider capabilities and preferences

## Scoring Components

### Core Scoring Factors

1. **Provider Type Bonus/Penalty** (+5.0 to -1000.0)
   - Massive penalties for wrong shift types (e.g., nocturnist taking day shift)
   - Bonuses for appropriate assignments

2. **Block Consistency** (+3.0 to -8.0)
   - Strong bonus for maintaining shift type consistency within blocks
   - Heavy penalties for mixing day and night shifts in same block

3. **Block Building** (+4.0 to -6.0)
   - Rewards building toward optimal block sizes
   - Penalties for standalone days and very short blocks

4. **Rest Requirements** (+1.0 to -10.0)
   - Strong penalties for inadequate rest between shifts
   - Bonuses for good rest patterns

5. **Workload Balance** (+2.0 to -4.0)
   - Encourages providers below target shift count
   - Penalties for exceeding expected shifts

6. **Weekend Distribution** (+3.0 to -2.0)
   - Ensures fair distribution of weekend coverage
   - Special bonuses for first weekend assignments

7. **Shift Preferences** (+2.0 to -5.0)
   - Respects provider shift type preferences
   - Penalties for assignments against preferences

8. **Timing Preferences** (+1.5 to -0.5)
   - Supports front-loaded vs back-loaded scheduling
   - Helps providers achieve desired work patterns

## Implementation Details

### Main Components

1. **ScheduleScorer Class** (`core/scoring.py`)
   - Central scoring engine
   - Provider-specific scoring methods
   - Comprehensive score calculation

2. **Enhanced Scheduler** (`core/scheduler.py`)
   - `assign_with_scoring()`: Main scoring-based assignment
   - `optimize_schedule_with_scoring()`: Post-assignment optimization
   - `calculate_total_schedule_score()`: Overall schedule evaluation

### Integration Points

The scoring system is integrated into the main scheduling flow:

1. **Initial Assignment**: Each shift assignment uses scoring to select the best provider
2. **Feasibility Filtering**: Only eligible providers are scored
3. **Score-Based Selection**: Highest scoring provider gets the assignment
4. **Randomization**: Close scores (within 10%) use randomization for variety
5. **Post-Assignment Optimization**: Swap-based optimization improves overall schedule

## Benefits Over Previous System

### 1. Predictable Behavior
- Clear scoring criteria replace heuristic-based decisions
- Consistent optimization across all assignment decisions

### 2. Work-Life Balance
- Block-based scheduling reduces provider fatigue
- Respects individual scheduling preferences
- Fair weekend and night shift distribution

### 3. Clinical Safety
- Prevents dangerous scheduling patterns
- Enforces adequate rest periods
- Maintains shift type consistency

### 4. Flexibility
- Easy to adjust scoring weights for different priorities
- Supports provider-specific preferences and constraints
- Handles multiple provider types with specialized rules

### 5. Optimization
- Built-in schedule optimization through scoring
- Can detect and fix scheduling violations
- Continuous improvement through iterative refinement

## Usage

The scoring system is automatically used when calling the main scheduling functions:

```python
# The generate_schedule function now uses scoring automatically
events = generate_schedule(year, month, providers, shift_types, 
                          shift_capacity, provider_rules, global_rules)
```

## Configuration

Scoring behavior can be tuned through:

- **Global Rules**: `RuleConfig` object controls system-wide constraints
- **Provider Rules**: Individual provider preferences and constraints
- **Scoring Weights**: Can be adjusted in the `ScheduleScorer` class methods

## Future Enhancements

Potential improvements to the scoring system:

1. **Machine Learning**: Learn optimal scoring weights from historical data
2. **Provider Feedback**: Incorporate satisfaction metrics into scoring
3. **Predictive Scoring**: Consider future schedule impacts
4. **Multi-Month Optimization**: Optimize across multiple months simultaneously
5. **Real-time Adjustment**: Dynamic scoring based on current schedule state

## Troubleshooting

If scheduling results are unsatisfactory:

1. Check provider rules and preferences
2. Verify global rule constraints
3. Review scoring logs for insights
4. Adjust scoring weights if needed
5. Use the validation output to identify specific issues
