from datetime import datetime, timezone
import pytest
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.resources import ConversationMessageContext, ConversationResourceReference
from career_agent.agent.middleware.argument_projection import project_atomic_arguments


@pytest.mark.parametrize('kind,reader,id_key', [
    ('job_research_report', 'get_job_research', 'report_id'),
    ('interview_preparation', 'get_interview_preparation', 'preparation_id'),
    ('mock_interview_report', 'get_mock_interview_result', 'mock_interview_report_id'),
])
def test_reference_resolution_retains_identity_and_type_without_selector_binding(kind, reader, id_key):
    reference = ConversationResourceReference(kind=kind, resource_id='private-resource', title='Same display title',
        **({'status_at_delivery': 'current', 'anchored_by_other_job': False} if kind == 'job_research_report' else {}))
    context = MainAgentContext(conversation_id='c1', profile=CareerProfileContext(user_id='u1'), user_message='Read this report',
        recent_messages=(ConversationMessageContext(role='assistant', content='Ready', created_at=datetime.now(timezone.utc), resource_refs=(reference,)),))
    handle = context.reference_handle(reference)
    projected = project_atomic_arguments(context, reader, {'reference': handle}, source_turn_id=None)
    assert 'private-resource' in projected.values()
    with pytest.raises(ValueError):
        project_atomic_arguments(context, reader, {'reference': 'unknown-reference'}, source_turn_id=None)
    with pytest.raises(ValueError):
        context.resolve_reference(reference=handle, kind='resume_version')


@pytest.mark.parametrize('reader', ['get_job_research', 'get_interview_preparation', 'get_mock_interview_result'])
def test_unbound_readback_is_rejected_by_existing_projection(reader):
    context = MainAgentContext(conversation_id='c1', profile=CareerProfileContext(user_id='u1'), user_message='Read it')
    with pytest.raises(ValueError):
        project_atomic_arguments(context, reader, {}, source_turn_id=None)
