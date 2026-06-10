```python
with open("observation.md", "r", encoding="utf-8") as f:
    obs_content = f.read()
print(obs_content)


```

```text
# Observations & Actions


## Raw observations

Environment exposes following `raw` observations:

- Ball information:
    - `ball` - [x, y, z] position of the ball.
    - `ball_direction` - [x, y, z] ball movement vector.
    - `ball_rotation` - [x, y, z] rotation angles in radians.
    - `ball_owned_team` - {-1, 0, 1}, -1 = ball not owned, 0 = left team, 1 = right team.
    - `ball_owned_player` - {0..N-1} integer denoting index of the player owning the ball.
- Left team:
    - `left_team` - N-elements vector with [x, y] positions of players.
    - `left_team_direction` - N-elements vector with [x, y] movement vectors of players.
    - `left_team_tired_factor` - N-elements vector of floats in the range {0..1}. 0 means player is not tired at all.
    - `left_team_yellow_card` - N-elements vector of integers denoting number of yellow cards a given player has (0 or 1).
    - `left_team_active` - N-elements vector of booleans denoting whether a given player is playing the game (False means player got a red card).
    - `left_team_roles` - N-elements vector denoting roles of players. The meaning is:
        - `0` = e_PlayerRole_GK - goalkeeper,
        - `1` = e_PlayerRole_CB - centre back,
        - `2` = e_PlayerRole_LB - left back,
        - `3` = e_PlayerRole_RB - right back,
        - `4` = e_PlayerRole_DM - defence midfield,
        - `5` = e_PlayerRole_CM - central midfield,
        - `6` = e_PlayerRole_LM - left midfield,
        - `7` = e_PlayerRole_RM - right midfield,
        - `8` = e_PlayerRole_AM - attack midfield,
        - `9` = e_PlayerRole_CF - central front,
- Right team:
    - `right_team` - same as for left team.
    - `right_team_direction` - same as for left team.
    - `right_team_tired_factor` - same as for left team.
    - `right_team_yellow_card` - same as for left team.
    - `right_team_active` - same as for left team.
    - `right_team_roles` - same as for left team.
- Controlled player information:
    - `active` - {0..N-1} integer denoting index of the controlled players.
    - `designated` - {0..N-1} integer denoting index of the designated player - the player leading the game, for example the one owning the ball. In non-multiagent mode it is always equal to `active`.
    - `sticky_actions` - 10-elements vectors of 0s or 1s denoting whether corresponding action is active:
        - `0` - `action_left`
        - `1` - `action_top_left`
        - `2` - `action_top`
        - `3` - `action_top_right`
        - `4` - `action_right`
        - `5` - `action_bottom_right`
        - `6` - `action_bottom`
        - `7` - `action_bottom_left`
        - `8` - `action_sprint`
        - `9` - `action_dribble`
- Match state:
    - `score` - pair of integers denoting number of goals for left and right teams, respectively.
    - `steps_left` - how many steps are left till the end of the match.
    - `game_mode` - current game mode, one of:
        - `0` = `e_GameMode_Normal`
        - `1` = `e_GameMode_KickOff`
        - `2` = `e_GameMode_GoalKick`
        - `3` = `e_GameMode_FreeKick`
        - `4` = `e_GameMode_Corner`
        - `5` = `e_GameMode_ThrowIn`
        - `6` = `e_GameMode_Penalty`
- Screen:
    - `frame` - three vectors of RGB pixels representing rendered
    screen. It is only exposed when rendering is enabled (`render` flag). Size
    of each vector is width by height of the rendered window, 1280 by 720 by default.

Where `N` is the number of players on the team.

*   Bottom left/right corner of the field is located at `[-1, 0.42]`
    and `[1, 0.42]`, respectively.
*   Top left/right corner of the field is located at `[-1, -0.42]`
    and `[1, -0.42]`, respectively.
*   Left/right goal is located at -1 and 1 X coordinate, respectively. They
    span between `-0.044` and `0.044` in Y coordinates.
*   Speed vectors represent a change in the position of the object within a
    single step.

In case of controlling `M` players (see multi-agent section), environment
returns a list of `M` observations, one per each controlled player. For
convenience, even if you control players on the right team, observations are
mirrored (center of your goal is at `[-1, 0]`).

## Observation Wrappers

In addition, environment provides wrappers which convert `raw` observations to a
different form:

### `simple115` (aka Simple115StateWrapper)

Simplified representation of a game state encoded with 115 floats. You can see
the code in `Simple115StateWrapper` for details.

WARNING: Note that this observation has a bug that when someone on a left team
gets a red card, positions of observations corresponding to the right team will
shift, which is very likely not a desirable behavior. We keep this wrapper for
backward compatibility reasons, but recommend to use simple115v2 instead.

### `simple115_v2`

Same as simple115, but with the bug fixed.

*   22 - (x,y) coordinates of left team players
*   22 - (x,y) direction of left team players
*   22 - (x,y) coordinates of right team players
*   22 - (x, y) direction of right team players
*   3 - (x, y and z) - ball position
*   3 - ball direction
*   3 - one hot encoding of ball ownership (noone, left, right)
*   11 - one hot encoding of which player is active
*   7 - one hot encoding of `game_mode`

Entries for players that are not active (either due to red cards or if number of
player is less than 11) are set to -1.

### `extracted` (aka SMMWrapper)

Simplified spatial (minimap) representation of a game state. It consists of
several 72 * 96 planes of bytes, filled in with 0s except for:

*   1st plane: 255s represent positions of players on the left team
*   2nd plane: 255s represent positions of players on the right team
*   3rd plane: 255s represent position of a ball
*   4th plane: 255s represent position of an active player

You can look at `generate_smm` method to see more details.

### `pixels`/`pixels_gray` (aka PixelsStateWrapper)

Pixel representation, downscaled to 72 * 96, and converted to a single grayscale
channel for `pixels_gray`.

In order to use this representation you have to enable rendering in
`create_environment` call (`render=True` parameter) or call `render` method on
the environment object (in case of not using `create_environment`).

### SingleAgentWrapper

For trivial integration with single agent/player learning algorithms we provide
a SingleAgentWrapper wrapper, that strips the first dimension of the
representation in the case of only one player being controlled.

## Actions

### Default action set

The default action set consists of 19 actions:

*   Idle actions

    *   `action_idle` = 0, a no-op action, sticky actions are not affected (player maintains his directional movement etc.).

*   Movement actions

    *   `action_left` = 1, run to the left, sticky action.
    *   `action_top_left` = 2, run to the top-left, sticky action.
    *   `action_top` = 3, run to the top, sticky action.
    *   `action_top_right` = 4, run to the top-right, sticky action.
    *   `action_right` = 5, run to the right, sticky action.
    *   `action_bottom_right` = 6, run to the bottom-right, sticky action.
    *   `action_bottom` = 7, run to the bottom, sticky action.
    *   `action_bottom_left` = 8, run to the bottom-left, sticky action.

*   Passing / Shooting

    *   `action_long_pass` = 9, perform a long pass to the player on your team. Player to pass the ball to is auto-determined based on the movement direction.
    *   `action_high_pass` = 10, perform a high pass, similar to `action_long_pass`.
    *   `action_short_pass` = 11, perform a short pass, similar to `action_long_pass`.
    *   `action_shot` = 12, perform a shot, always in the direction of the opponent's goal.

*   Other actions

    *   `action_sprint` = 13, start sprinting, sticky action. Player moves faster, but has worse ball handling.
    *   `action_release_direction` = 14, reset current movement direction.
    *   `action_release_sprint` = 15, stop sprinting.
    *   `action_sliding` = 16, perform a slide (effective when not having a ball).
    *   `action_dribble` = 17, start dribbling (effective when having a ball), sticky action. Player moves slower, but it is harder to take over the ball from him.
    *   `action_release_dribble` = 18, stop dribbling.

### V2 action set

It is an extension of the default action set:

*   `action_builtin_ai` = 19, let game's built-in AI generate an action.



```

Hành động thứ 19 (`action_builtin_ai = 19`) trong Google Research Football (GRF) có chức năng: **"let game's built-in AI generate an action"** (Cho phép AI tích hợp sẵn của trò chơi tự động tạo ra hành động).

Hành động này có thể là một "vũ khí bí mật" cực kỳ mạnh mẽ để hỗ trợ huấn luyện, đặc biệt là trong kiến trúc phân cấp HES-COMA của bạn. Dưới đây là phân tích chi tiết về cách nó có thể hỗ trợ từng Phase:

### 1. Ứng dụng trong Phase 1 (GAgent - Chiếm lĩnh không gian)

Trong Phase 1, mục tiêu của 10 cầu thủ (trừ người đá phạt) là chạy chỗ để chiếm lĩnh các "hố năng lượng" trống trải. Hiện tại, chúng ta đang để AI (mạng nơ-ron) tự mò mẫm từ đầu (từ con số 0) để học cách di chuyển bằng cách thử và sai thông qua các hành động `0-8` hoặc `14` (đổi hướng, giữ nguyên, dừng lại).

**Sử dụng `action_builtin_ai` để Khởi động ấm (Warm-start / Imitation Learning):**
AI tích hợp sẵn của GRF (Bot của game) đã được lập trình sẵn bằng hệ thống luật (rule-based) để biết cách chạy chỗ tương đối cơ bản. Bạn có thể sử dụng hành động 19 để tăng tốc độ hội tụ ở những ván (episodes) đầu tiên:

* **Phương pháp Epsilon-Greedy kết hợp Built-in AI:**
Thay vì chọn hành động ngẫu nhiên hoàn toàn (Random Exploration) khi $\epsilon$ cao trong giai đoạn đầu, hãy để xác suất $\epsilon$ đó chọn `action_builtin_ai = 19`.
* *Cách thức:* Khi Agent thăm dò, nó sẽ chạy theo hướng mà Bot của game cho là hợp lý. Những quỹ đạo (trajectories) do Bot tạo ra này sẽ đưa cầu thủ đến các vị trí di chuyển có ý nghĩa hơn là chạy loạn xạ đâm vào tường.
* *Lợi ích:* Mạng Critic của bạn sẽ nhanh chóng nhận được các mẫu dữ liệu (samples) có điểm thưởng Năng lượng (`rewards_energy`) cao (vì Bot thường tránh hậu vệ và tìm đến bóng). Điều này giúp Mạng Actor học được một chính sách (Policy) "có hình hài" chỉ trong vài trăm episodes thay vì vài ngàn. Khi $\epsilon$ giảm dần, GAgent sẽ tự tin thay thế Bot và tìm ra các đường chạy cắt mặt đột biến (dựa trên hố năng lượng sâu của bạn) mà bản thân Bot mặc định không biết làm.



### 2. Ứng dụng trong Phase 2 (LAgent - Kỹ thuật cục bộ)

Trong Phase 2, khi một cầu thủ nhận được bóng (GAgent chọn lệnh STOP), LAgent sẽ thức tỉnh để quyết định sút, chuyền ngắn, chuyền dài, v.v.

**Sử dụng `action_builtin_ai` làm "Giáo viên hướng dẫn sút" (Teacher / Baseline Expert):**

* **Vấn đề hiện tại:** LAgent phải học cách chọn đúng thời điểm để gọi lệnh `action_shot` (12) hoặc `action_high_pass` (10). Thường thì nó sẽ sút bậy ra ngoài hoặc chuyền hỏng rất nhiều lần trước khi ghi được 1 bàn thắng. Phần thưởng Ghi bàn quá thưa thớt khiến nó nản.
* **Giải pháp với Built-in AI:**
Bạn có thể thiết lập một số ván đấu ở Phase 2 (ví dụ: 10% số ván) ép LAgent dùng lệnh `19`.
* Trong những ván này, Bot của GRF sẽ tự động lấy bóng và thực hiện một cú sút bồi hoặc đánh đầu (nếu nó thấy cơ hội).
* Mặc dù mạng LAgent không tự quyết định, nhưng nó **VẪN ĐƯỢC QUAN SÁT** (nhận Obs) và **NHẬN ĐIỂM THƯỞNG** (từ cú sút thành công của Bot).
* Thông qua bộ đệm (Replay Buffer), Mạng Critic của LAgent sẽ học được hàm Giá trị $Q(s, a)$ rằng: *"À, ở cái góc hẹp này, khoảng cách này, nếu có hành động sút được thực thi, khả năng cao là sẽ có bàn thắng (+1)"*.
* Khi LAgent quay lại tự đá (tự chọn Action 12), nó đã có sẵn một hàm $Q$ vững chắc định hướng, giúp nó tự tin dứt điểm thay vì sợ hãi chuyền ngược về sân nhà.



### Tóm lại: Có nên tích hợp không?

**CÓ, đặc biệt là trong giai đoạn đầu (Pre-training) của Phase 1.**

Việc đưa `action_builtin_ai` (19) vào làm **Guided Exploration** (Khám phá có hướng dẫn) sẽ khắc phục điểm yếu lớn nhất của RL là quá trình khởi động quá chậm. Tuy nhiên, bạn KHÔNG nạp action 19 vào không gian hành động đầu ra của mạng Nơ-ron. Mạng Nơ-ron vẫn chỉ xuất ra 10 hành động. Hành động 19 chỉ được can thiệp ở tầng code Môi trường (Wrapper) hoặc thuật toán chọn hành động (Epsilon-greedy) trong file `train_phase1.py` để "cầm tay chỉ việc" cho các Agent trong những giờ huấn luyện đầu tiên.