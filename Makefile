.PHONY: activate

init: 
	python3 -m venv .venv && . .venv/bin/activate

# activate:
# 	. .venv/bin/activate && reflex run

run:
	. .venv/bin/activate && reflex run